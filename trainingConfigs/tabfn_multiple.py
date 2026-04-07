from functools import partial
import numpy as np
import sklearn.datasets
import torch
import shap
from torch.utils.data import random_split, TensorDataset, DataLoader
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from tabpfn import TabPFNClassifier
#from tabpfn.finetune_utils import clone_model_for_evaluation
#from tabpfn.utils import meta_dataset_collator
from tqdm import trange, tqdm
import time
import numpy as np
from sklearn.calibration import LabelEncoder
from modelConfigs.tabfnModules import TabPFNForSHAP, TabPFNModelWithFSL
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import torch
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
import pandas as pd
from torch import nn
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from feature_position_walk import generate_feature_position_walk_plot
from trainingConfigs.execution_store import PretrainedExecutionStore
from trainingTestStep import get_weighted_cross_entropy_loss, trainingModule
from utils import LogPrinter, calculate_kruskal_dunn_2, calculate_kruskal_dunn_3, calculate_kruskal_dunn_7, calculate_normalized_weights, generate_execution_id, displayTopFeatures, find_normalized_weights, get_feature_rankings, get_feature_weights_as_numpy, persist_wtsne_input, calculate_aggregated_feature_importance
from metrics import Selection_Accuracy, jaccard_similarity, pearson_correlation, spearman_correlation
from wtsne import WTSNEv2
from evaluation import calculate_prediction_metrics
from data.loadDataset import folds_to_dataloaders, numpy_to_dataloaders
from featureSelectionLayer import compare_model_weights, freezeParams, fs_layer_regularization, transfer_weights
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
device = "cuda" if torch.cuda.is_available() else "cpu"


def tabfn_multiple_training(name):
    shap_size = 250 
    shap_size_representative = 10 # Consumes too much memory
    epochs = 10
    seed = 42
    regularization_l = 0.0025
    learning_rate = 0.001
    test_percentage=0.1
    batch_size=16
    dataset_path="data/synthetic/synth_3000samples_100features_30informative.csv"
    ignored_columns=[]
    label_column='class'
    should_persist=True
    weight_scaler=MinMaxScaler
    should_train_posthoc_fsl=False
    
    def prepare_data():
        logger.log_text("--- 1. Data Preparation ---")
        df = pd.read_csv(dataset_path)    
        df = df.drop(ignored_columns, axis=1)
        feature_columns = list(filter(lambda x: x != label_column, df.columns))
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])
        logger.log_text(str(dict(zip(le.classes_, le.transform(le.classes_)))), filename="label_mapping.txt")
        logger.log_text("Splitting dataset...")
        X = df[feature_columns]
        y = df[label_column]
        X = torch.Tensor(X.to_numpy()).type(torch.float32).to(device)
        y = torch.Tensor(y.to_numpy()).type(torch.long).to(device)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=42, stratify=y)
        logger.log_text(f"Train samples: {X_train.shape[0]}.")
        logger.log_text(f"Test samples: {X_test.shape[0]}.")
        return X, y, X_train, X_test, y_train, y_test

    def setup_model(X, y, seed) -> tuple[TabPFNClassifier, Optimizer, dict]:
        """Initializes the TabPFN classifier, optimizer, and training configs."""
        print("--- 2. Model and Optimizer Setup ---")
        classifier_config = {
            "ignore_pretraining_limits": True,
            "device": device,
            "n_estimators": 2,
            "random_state": seed,
            "inference_precision": torch.float32,
        }
        classifier = TabPFNClassifier(
            **classifier_config, differentiable_input=True
        )
        classifier.fit(X, y)
        return classifier

    # Generate the execution id and folders

    execution_id = generate_execution_id(name, persist=should_persist)

    # Create the main logger

    logger = LogPrinter(name, execution_id, persist=should_persist)
    logger.log_text(f"Starting execution: {execution_id}.")

    # --- Setup Data, Model, and Dataloader ---
    X, y, X_train, X_test, y_train, y_test = prepare_data()

    if should_train_posthoc_fsl:
        try:
            pfn_classifier = setup_model(X_train, y_train, seed)
            classifier = TabPFNModelWithFSL(pfn_classifier)
            optimizer = AdamW(classifier.fs.parameters(), lr=learning_rate)
            train_dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(train_dataset, batch_size=batch_size)
            loss_function = torch.nn.CrossEntropyLoss()

            # --- Finetuning and Evaluation Loop ---
            
            logger.log_text("--- 3. Starting Finetuning & Evaluation ---")
            for epoch in trange(epochs + 1, desc="Epochs"):
                if epoch == 0:
                    continue

                for X_train_batch, y_train_batch in tqdm(
                    dataloader,
                    desc=f"[Epoch {epoch}/{epochs}]",
                    unit="batch",
                    leave=False
                ):
                    optimizer.zero_grad()
                    predictions = classifier(X_train_batch)
                    loss = loss_function(
                        predictions.permute(0, 2, 1).squeeze(),
                        y_train_batch
                    )
                    loss += fs_layer_regularization(classifier, l=regularization_l)
                    loss.backward()
                    optimizer.step()

            predictions = classifier.forward(X_test)
            predictions = predictions.permute(0, 2, 1).squeeze()
            loss = loss_function(predictions, y_test)
            f1 = f1_score(y_test.detach().cpu().numpy(), predictions.argmax(axis=1))
            logger.log_text(f"F1 score: {f1}")
            
            try:
                silhouette_without_weights = WTSNEv2(logger, X.detach().cpu().numpy(), y.detach().cpu().numpy(), name="without weights", seed=seed, weight_scaler=weight_scaler)
                logger.log_text(str(silhouette_without_weights))
            except Exception as e:
                print(e)
                print(silhouette_without_weights)
            
            try:
                silhouette_with_weights = WTSNEv2(logger, X.detach().cpu().numpy(), y.detach().cpu().numpy(), model=classifier, name="with Post-hoc FSL weights", seed=seed, weight_scaler=weight_scaler)
                logger.log_text(f"Posthoc silhouette: {str(silhouette_with_weights)}")
            except Exception as e:
                print(e)
                print(silhouette_with_weights)

            try:
                logger.log_np_array(classifier.fs.get_activated_weights().detach().cpu().numpy(), "posthoc_weights", fmt='%f')
            except Exception as e:
                print(e)
                print(classifier.fs.get_activated_weights().detach().cpu().numpy())
        except Exception as e:
            print(e)

    try:
        pfn_classifier = setup_model(X_train, y_train, seed)
        classifier = TabPFNForSHAP(pfn_classifier)
        shap_samples = shap.sample(X_test.detach().cpu().numpy(), shap_size)
        shap_representative = shap.sample(X_train.detach().cpu().numpy(), shap_size_representative)
        explainer = shap.KernelExplainer(model=classifier, data=shap_representative)
        shap_values = explainer.shap_values(shap_samples)
        shap_values = np.abs(shap_values).mean(0)
        n_labels = len(shap_values[0])
        transpose = []
        for i in range(0, n_labels):
            transpose.append([])
        for feature_per_class in shap_values:
            for i in range(0, n_labels):
                transpose[i].append(feature_per_class[i])
        for i in range(0, n_labels):
            transpose[i] = np.array(transpose[i], dtype=np.float64)
        shap_values = transpose
        shap_values = np.max(shap_values, axis=0)
        logger.log_np_array(shap_values, "shap_weights", fmt='%f')
        shap_silhouette_with_weights = WTSNEv2(logger, X.detach().cpu().numpy(), y.detach().cpu().numpy(), weights=torch.from_numpy(shap_values), name="with SHAP weights", seed=seed, weight_scaler=weight_scaler)
        logger.log_text(f"SHAP silhouette: {str(shap_silhouette_with_weights)}")
    except Exception as e:
        print(e)

    try:
        ig = IntegratedGradients(pfn_classifier)
        integrated_gradients_attributes = ig.attribute(X_test, n_steps=50)
        integrated_gradients_feature_weights = torch.mean(torch.abs(integrated_gradients_attributes), dim=0)
        logger.log_np_array(integrated_gradients_feature_weights.detach().cpu().numpy(), "integrated_gradients_weights", fmt='%f')
        integrated_gradients_silhouette_with_weights = WTSNEv2(logger, X.detach().cpu().numpy(), y.detach().cpu().numpy(), weights=integrated_gradients_feature_weights, name="with Integrated Dradients weights", seed=seed, weight_scaler=weight_scaler)
        logger.log_text(f"Integrated Gradients silhouette: {str(integrated_gradients_silhouette_with_weights)}")
    except Exception as e:
        print(e)

    try:
        ig_nt = NoiseTunnel(ig)
        noise_tunnel_attributes = ig_nt.attribute(X_test)
        noise_tunnel_feature_weights = torch.mean(torch.abs(noise_tunnel_attributes), dim=0)
        logger.log_np_array(noise_tunnel_feature_weights.detach().cpu().numpy(), "noise_tunnel_weights", fmt='%f')
        noise_tunnel_silhouette_with_weights = WTSNEv2(logger, X.detach().cpu().numpy(), y.detach().cpu().numpy(), weights=noise_tunnel_feature_weights, name="with Noise Tunnel weights", seed=seed, weight_scaler=weight_scaler)
        logger.log_text(f"Noise Tunnel silhouette: {str(noise_tunnel_silhouette_with_weights)}")
    except Exception as e:
        print(e)

    try:
        dl = DeepLift(pfn_classifier)
        deep_lift_attributes = dl.attribute(X_test)
        deep_lift_feature_weights = torch.mean(torch.abs(deep_lift_attributes), dim=0)
        logger.log_np_array(deep_lift_feature_weights.detach().cpu().numpy(), "deep_lift_weights", fmt='%f')
        deep_lift_silhouette_with_weights = WTSNEv2(logger, X.detach().cpu().numpy(), y.detach().cpu().numpy(), weights=deep_lift_feature_weights, name="with DeepLift weights", seed=seed, weight_scaler=weight_scaler)
        logger.log_text(f"DeepLift silhouette: {str(deep_lift_silhouette_with_weights)}")
    except Exception as e:
        print(e)

    try:
        gs = GradientShap(pfn_classifier)
        gradient_shap_attributes = gs.attribute(X_test, X_train)
        gradient_shap_feature_weights = torch.mean(torch.abs(gradient_shap_attributes), dim=0)
        logger.log_np_array(gradient_shap_feature_weights.detach().cpu().numpy(), "gradient_shap_weights", fmt='%f')
        gradient_shap_silhouette_with_weights = WTSNEv2(logger, X.detach().cpu().numpy(), y.detach().cpu().numpy(), weights=gradient_shap_feature_weights, name="with Gradient Shap weights", seed=seed, weight_scaler=weight_scaler)
        logger.log_text(f"Gradient Shap silhouette: {str(gradient_shap_silhouette_with_weights)}")
    except Exception as e:
        print(e)

    try:
        fa = FeatureAblation(pfn_classifier)
        feature_ablation_attributes = fa.attribute(X_test)
        feature_ablation_feature_weights = torch.mean(torch.abs(feature_ablation_attributes), dim=0)
        logger.log_np_array(feature_ablation_feature_weights.detach().cpu().numpy(), "feature_ablation_weights", fmt='%f')
        feature_ablation_silhouette_with_weights = WTSNEv2(logger, X.detach().cpu().numpy(), y.detach().cpu().numpy(), weights=feature_ablation_feature_weights, name="with Feature Ablation weights", seed=seed, weight_scaler=weight_scaler)
        logger.log_text(f"Feature Ablation silhouette: {str(feature_ablation_silhouette_with_weights)}")
    except Exception as e:
        print(e)

    
    logger.log_text("--- ✅ Finetuning Finished ---")