import time
import numpy as np
import shap
import torch
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from feature_position_walk import generate_feature_position_walk_plot
from trainingConfigs.execution_store import ExecutionStore
from trainingTestStep import trainingModule
from utils import LogPrinter, calculate_kruskal_dunn_3, calculate_kruskal_dunn_5, calculate_normalized_weights, generate_execution_id, displayTopFeatures, find_normalized_weights, get_feature_rankings, get_feature_weights_as_numpy, get_weight_per_class_from_shap, persist_wtsne_input, calculate_aggregated_feature_importance
from data.loadDataset import folds_to_dataloaders, numpy_to_dataloaders
from metrics import Selection_Accuracy, jaccard_similarity, pearson_correlation, silhouetteMetric, spearman_correlation
from wtsne import WTSNEv2
from evaluation import calculate_prediction_metrics
from featureSelectionLayer import freezeParams, transfer_weights
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
device = "cuda" if torch.cuda.is_available() else "cpu"



def multiple_training(name, base_model, model_with_fsl, dataset_path, label_column, has_numeric_labels=True, ignored_columns=[], num_of_tests=3, test_percentage=0.1, seed=None, batch_size=32, n_epochs_base=50, n_epochs_fsl=50, 
                      n_epochs_fsl_posthoc=50, learning_rate=0.01, should_persist=True, num_of_informative_features_to_display=10, jaccard_k_list=None, l=0.001, scaler=StandardScaler, shap_k=500, shap_representative_k=500, general_weights_from_absolute_values=True):
    general_start_time = time.perf_counter()

    # Models

    Model = base_model
    ModelWithFSL = model_with_fsl
    
    # Generate the execution id and folders

    execution_id = generate_execution_id(name, persist=should_persist)

    # Create the main logger

    logger = LogPrinter(name, execution_id, persist=should_persist)
    logger.log_text(f"Starting execution: {execution_id}.")

    # Log configuration

    logger.log_text(f"Data: {dataset_path}.")
    logger.log_text(f"Using seed: {seed}.")
    logger.log_text(f"Number of tests: {num_of_tests}.")
    logger.log_text(f"Test percentage: {test_percentage}.")

    # Setup torch seed

    logger.log_text(f"Setting up seed {seed}")
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # Read the dataset

    logger.log_text("Loading dataset...")
    df = pd.read_csv(dataset_path)    
    
    df = df.drop(ignored_columns, axis=1)
    feature_columns = list(filter(lambda x: x != label_column, df.columns))
    informative_features = [col for col in feature_columns if 'informative' in col]

    # Prepare label columns

    if not has_numeric_labels:
        logger.log_text("Converting label values to number")
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])
        logger.log_text(str(dict(zip(le.classes_, le.transform(le.classes_)))), filename="label_mapping.txt")

    # Split the dataset into train and test

    logger.log_text("Splitting dataset...")
    X = df[feature_columns]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=seed, stratify=y)

    logger.log_text(f"Train samples: {X_train.shape[0]}.")
    logger.log_text(f"Test samples: {X_test.shape[0]}.")

    # Create test dataloader

    logger.log_text("Creating test dataloader...")

    X_test = pd.DataFrame(scaler().fit_transform(X_test), columns=X.columns) # StandardScaler
    X_test_tensor = torch.from_numpy(X_test.to_numpy()).type(torch.float32).to(device)

    test_dataloader = numpy_to_dataloaders(X_test, y_test, batch_size=batch_size)

    # Create KFolds

    logger.log_text("Creating folds...")
    skf = StratifiedKFold(n_splits=num_of_tests, shuffle=True, random_state=seed)
    folds = list(skf.split(X_train, y_train))
    logger.log_text(f"Number of folds created: {len(folds)}.")
    for i, (train_idx, val_idx) in enumerate(folds):
        logger.log_text(f"Fold {i + 1}: {len(train_idx)} training samples, {len(val_idx)} validation samples.")

    # Create store to persist fold results

    store = ExecutionStore()

    # Start the training for each fold

    logger.log_text("Starting folds training...")
    total_training_start_time = time.perf_counter()

    for fold_index, (train_index, val_index) in enumerate(folds):

        # Generating fold execution id 
        
        internal_execution_id = generate_execution_id(name, external_id=execution_id, index=fold_index, persist=should_persist)

        # Generate fold logger

        fold_logger = LogPrinter(name, internal_execution_id, persist=should_persist, father=logger)
        fold_logger.log_text(f"Starting fold {fold_index + 1}/{num_of_tests}.")

        # Prepare fold dataloaders

        fold_logger.log_text("Preparing fold dataloaders...")
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]
        X_fold_train = pd.DataFrame(scaler().fit_transform(X_fold_train), columns=X.columns)
        X_fold_val = pd.DataFrame(scaler().fit_transform(X_fold_val), columns=X.columns)
        X_fold_train_tensor = torch.from_numpy(X_fold_train.to_numpy()).type(torch.float32).to(device)
        X_fold_val_tensor = torch.from_numpy(X_fold_val.to_numpy()).type(torch.float32).to(device)
        train_dataloader, val_dataloader = folds_to_dataloaders(X_fold_train, y_fold_train, X_fold_val, y_fold_val, batch_size=batch_size)
        fold_logger.log_text(f"Fold {fold_index + 1} - Train samples: {X_fold_train.shape[0]}.")
        fold_logger.log_text(f"Fold {fold_index + 1} - Validation samples: {X_fold_val.shape[0]}.")

        # Train three models: without FSL, with FSL, with FSL post-hoc and persist the execution times

        fold_logger.log_text("Training model without FSL.")
        model_without_fsl = Model()
        start_time = time.perf_counter()
        model_without_fsl = trainingModule(model_without_fsl, train_dataloader, val_dataloader, n_epochs=n_epochs_base, 
                                         earlyStop=False, isFSLpresent=False, seed=seed, print_function=fold_logger.log_text, lr=learning_rate, l=l)
        elapsed = time.perf_counter() - start_time
        store.training_time_without_weights.append(elapsed)
        
        fold_logger.log_text("Training model with FSL.")
        model_with_fsl = ModelWithFSL(name="Model with FSL")
        start_time = time.perf_counter()
        model_with_fsl = trainingModule(model_with_fsl, train_dataloader, val_dataloader, n_epochs=n_epochs_fsl, 
                                        earlyStop=False, isFSLpresent=True, seed=seed, print_function=fold_logger.log_text, lr=learning_rate, l=l)
        elapsed = time.perf_counter() - start_time
        store.training_time_with_fsl.append(elapsed)

        fold_logger.log_text("Training model with Post-hoc FSL.")
        model_with_fsl_posthoc = ModelWithFSL(name="Model with Post-hoc FSL")
        transfer_weights(model_without_fsl, model_with_fsl_posthoc)
        freezeParams(model_with_fsl_posthoc)
        start_time = time.perf_counter()
        model_with_fsl_posthoc = trainingModule(model_with_fsl_posthoc, train_dataloader, val_dataloader, n_epochs=n_epochs_fsl_posthoc, 
                                                earlyStop=False, isFSLpresent=True, seed=seed, print_function=fold_logger.log_text, lr=learning_rate, l=l)
        elapsed = time.perf_counter() - start_time
        store.training_time_with_fsl_posthoc.append(elapsed)

        # Apply others posthoc methods
        '''
        ig = IntegratedGradients(model_without_fsl)
        ig_nt = NoiseTunnel(ig)
        dl = DeepLift(model_without_fsl)
        gs = GradientShap(model_without_fsl)
        fa = FeatureAblation(model_without_fsl)

        integrated_gradients_attributes = ig.attribute(X_test_tensor, n_steps=50)
        noise_tunnel_attributes = ig_nt.attribute(X_test_tensor)
        deep_lift_attributes = dl.attribute(X_test_tensor)
        gradient_shap_attributes = gs.attribute(X_test_tensor, X_fold_train_tensor)
        feature_ablation_attributes = fa.attribute(X_test_tensor)

        if general_weights_from_absolute_values:
            integrated_gradients_feature_weights = torch.mean(torch.abs(integrated_gradients_attributes), dim=0)
            noise_tunnel_feature_weights = torch.mean(torch.abs(noise_tunnel_attributes), dim=0)
            deep_lift_feature_weights = torch.mean(torch.abs(deep_lift_attributes), dim=0)
            gradient_shap_feature_weights = torch.mean(torch.abs(gradient_shap_attributes), dim=0)
            feature_ablation_feature_weights = torch.mean(torch.abs(feature_ablation_attributes), dim=0)
        else:
            integrated_gradients_feature_weights = calculate_aggregated_feature_importance(integrated_gradients_attributes)
            noise_tunnel_feature_weights = calculate_aggregated_feature_importance(noise_tunnel_attributes)
            deep_lift_feature_weights = calculate_aggregated_feature_importance(deep_lift_attributes)
            gradient_shap_feature_weights = calculate_aggregated_feature_importance(gradient_shap_attributes)
            feature_ablation_feature_weights = calculate_aggregated_feature_importance(feature_ablation_attributes)

        displayTopFeatures(None, feature_columns, feature_weights=integrated_gradients_feature_weights, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-integrated-gradients.txt"))
        displayTopFeatures(None, feature_columns, feature_weights=noise_tunnel_feature_weights, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-noise-tunnel.txt"))
        displayTopFeatures(None, feature_columns, feature_weights=deep_lift_feature_weights, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-deep-lift.txt"))
        displayTopFeatures(None, feature_columns, feature_weights=gradient_shap_feature_weights, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-gradient-shap.txt"))
        displayTopFeatures(None, feature_columns, feature_weights=feature_ablation_feature_weights, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-feature-ablation.txt"))
        '''
        displayTopFeatures(model_with_fsl, feature_columns, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-fsl.txt"))
        displayTopFeatures(model_with_fsl_posthoc, feature_columns, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-posthoc-fsl.txt"))

        # Persist weights as original wtsne input
        '''
        persist_wtsne_input(None, feature_columns, feature_weights=integrated_gradients_feature_weights, name="integrated-gradients", logger=fold_logger)
        persist_wtsne_input(None, feature_columns, feature_weights=noise_tunnel_feature_weights, name="noise-tunnel", logger=fold_logger)
        persist_wtsne_input(None, feature_columns, feature_weights=deep_lift_feature_weights, name="deep-lift", logger=fold_logger)
        persist_wtsne_input(None, feature_columns, feature_weights=gradient_shap_feature_weights, name="gradient-shap", logger=fold_logger)
        persist_wtsne_input(None, feature_columns, feature_weights=feature_ablation_feature_weights, name="feature-ablation", logger=fold_logger)
        '''
        persist_wtsne_input(model_with_fsl, feature_columns, name="fsl", logger=fold_logger)
        persist_wtsne_input(model_with_fsl_posthoc, feature_columns, name="fsl-posthoc", logger=fold_logger)

        # Save feature weights 
        '''
        integrated_gradients_feature_weights_np = integrated_gradients_feature_weights.squeeze().detach().cpu().numpy()
        noise_tunnel_feature_weights_np = noise_tunnel_feature_weights.squeeze().detach().cpu().numpy()
        deep_lift_feature_weights_np = deep_lift_feature_weights.squeeze().detach().cpu().numpy()
        gradient_shap_feature_weights_np = gradient_shap_feature_weights.squeeze().detach().cpu().numpy()
        feature_ablation_feature_weights_np = feature_ablation_feature_weights.squeeze().detach().cpu().numpy()
        store.feature_weights_integrated_gradients.append(integrated_gradients_feature_weights_np)
        store.feature_weights_noise_tunnel.append(noise_tunnel_feature_weights_np)
        store.feature_weights_deep_lift.append(deep_lift_feature_weights_np)
        store.feature_weights_gradient_shap.append(gradient_shap_feature_weights_np)
        store.feature_weights_feature_ablation.append(feature_ablation_feature_weights_np)
        '''
        weights_with_fsl = get_feature_weights_as_numpy(model_with_fsl)
        store.feature_weights_fsl.append(weights_with_fsl)
        weights_with_fsl_posthoc = get_feature_weights_as_numpy(model_with_fsl_posthoc)
        store.feature_weights_fsl_posthoc.append(weights_with_fsl_posthoc)

        # Persist feature weights

        fold_logger.log_text("Persisting feature weights...")
        '''
        fold_logger.log_np_array(integrated_gradients_feature_weights_np, filename=f"integrated_gradients_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(noise_tunnel_feature_weights_np, filename=f"noise_tunnel_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(deep_lift_feature_weights_np, filename=f"deep_lift_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(gradient_shap_feature_weights_np, filename=f"gradient_shap_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(feature_ablation_feature_weights_np, filename=f"feature_ablation_feature_weights.txt", fmt='%f')
        '''
        fold_logger.log_np_array(weights_with_fsl, filename=f"fsl_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(weights_with_fsl_posthoc, filename=f"fsl_posthoc_feature_weights.txt", fmt='%f')
    
        # Get normalized weights for the two models with FSL
        '''
        integrated_gradients_normalized_feature_weights = calculate_normalized_weights(integrated_gradients_feature_weights)
        noise_tunnel_normalized_feature_weights = calculate_normalized_weights(noise_tunnel_feature_weights)
        deep_lift_normalized_feature_weights = calculate_normalized_weights(deep_lift_feature_weights)
        gradient_shap_normalized_feature_weights = calculate_normalized_weights(gradient_shap_feature_weights)
        feature_ablation_normalized_feature_weights = calculate_normalized_weights(feature_ablation_feature_weights)
        '''
        fsl_normalized_feature_weights = find_normalized_weights(model_with_fsl)
        fsl_posthoc_normalized_feature_weights = find_normalized_weights(model_with_fsl_posthoc)

        # Persist normalized feature weights for the two models with FSL

        fold_logger.log_text("Persisting normalized feature weights...")
        '''
        fold_logger.log_np_array(integrated_gradients_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"integrated_gradients_normalized_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(noise_tunnel_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"noise_tunnel_normalized_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(deep_lift_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"deep_lift_normalized_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(gradient_shap_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"gradient_shap_normalized_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(feature_ablation_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"feature_ablation_normalized_feature_weights.txt", fmt='%f')
        '''
        fold_logger.log_np_array(fsl_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"fsl_normalized_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(fsl_posthoc_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"fsl_posthoc_normalized_feature_weights.txt", fmt='%f')

        # Persist feature rankings for the two models with FSL

        fold_logger.log_text("Persisting feature rankings...")
        '''
        fold_logger.log_np_array(get_feature_rankings(None, feature_columns, weights=integrated_gradients_feature_weights), filename=f"integrated_gradients_feature_rankings.txt", fmt='%s')
        fold_logger.log_np_array(get_feature_rankings(None, feature_columns, weights=noise_tunnel_feature_weights), filename=f"noise_tunnel_feature_rankings.txt", fmt='%s')
        fold_logger.log_np_array(get_feature_rankings(None, feature_columns, weights=deep_lift_feature_weights), filename=f"deep_lift_feature_rankings.txt", fmt='%s')
        fold_logger.log_np_array(get_feature_rankings(None, feature_columns, weights=gradient_shap_feature_weights), filename=f"gradient_shap_feature_rankings.txt", fmt='%s')
        fold_logger.log_np_array(get_feature_rankings(None, feature_columns, weights=feature_ablation_feature_weights), filename=f"feature_ablation_feature_rankings.txt", fmt='%s')
        '''
        fold_logger.log_np_array(get_feature_rankings(model_with_fsl, feature_columns), filename=f"fsl_feature_rankings.txt", fmt='%s')
        fold_logger.log_np_array(get_feature_rankings(model_with_fsl_posthoc, feature_columns), filename=f"fsl_posthoc_feature_rankings.txt", fmt='%s')

        # Calculate feature selection metrics

        if len(informative_features) > 0:
            fold_logger.log_text("Calculating feature selection metrics...")
            '''
            pifs, psfi = Selection_Accuracy(integrated_gradients_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_with_integrated_gradients.append(pifs)
            store.psfi_with_integrated_gradients.append(psfi)

            pifs, psfi = Selection_Accuracy(noise_tunnel_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_with_noise_tunnel.append(pifs)
            store.psfi_with_noise_tunnel.append(psfi)

            pifs, psfi = Selection_Accuracy(deep_lift_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_with_deep_lift.append(pifs)
            store.psfi_with_deep_lift.append(psfi)

            pifs, psfi = Selection_Accuracy(gradient_shap_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_with_gradient_shap.append(pifs)
            store.psfi_with_gradient_shap.append(psfi)

            pifs, psfi = Selection_Accuracy(feature_ablation_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_with_feature_ablation.append(pifs)
            store.psfi_with_feature_ablation.append(psfi)
            '''
            pifs, psfi = Selection_Accuracy(fsl_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_with_fsl.append(pifs)
            store.psfi_with_fsl.append(psfi)

            pifs, psfi = Selection_Accuracy(fsl_posthoc_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_with_fsl_posthoc.append(pifs)
            store.psfi_with_fsl_posthoc.append(psfi)
        else:
            fold_logger.log_text("No informative features found, skipping feature selection metrics.")

        # Calculate prediction metrics for the three models

        fold_logger.log_text("Calculating prediction metrics for model without FSL.")
        f1, acc, precision, recall = calculate_prediction_metrics(fold_logger, model_without_fsl, test_dataloader, print_function=fold_logger.log_text, name="without FSL")
        store.f1_scores_without_weights.append(f1)
        store.accuracy_without_weights.append(acc)
        store.precision_without_weights.append(precision)
        store.recall_without_weights.append(recall)

        fold_logger.log_text("Calculating prediction metrics for model with FSL.")
        f1, acc, precision, recall = calculate_prediction_metrics(fold_logger, model_with_fsl, test_dataloader, print_function=fold_logger.log_text, name="with FSL")
        store.f1_scores_with_fsl.append(f1)
        store.accuracy_with_fsl.append(acc)
        store.precision_with_fsl.append(precision)
        store.recall_with_fsl.append(recall)

        fold_logger.log_text("Calculating prediction metrics for model with Post-hoc FSL.")
        f1, acc, precision, recall = calculate_prediction_metrics(fold_logger, model_with_fsl_posthoc, test_dataloader, print_function=fold_logger.log_text, name="with Post-hoc FSL")
        store.f1_scores_with_fsl_posthoc.append(f1)
        store.accuracy_with_fsl_posthoc.append(acc)
        store.precision_with_fsl_posthoc.append(precision)
        store.recall_with_fsl_posthoc.append(recall)

        # Calculate weighted t-SNE and silhouette

        fold_logger.log_text("Calculating standart t-SNE and silhouette without weights.")
        silhouette_without_weights = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), name="without weights")
        store.silhouette_without_weights.append(silhouette_without_weights)
        '''
        fold_logger.log_text("Calculating weighted t-SNE and silhouette for Integrated Gradients.")
        silhouette_with_integrated_gradients = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), weights=integrated_gradients_feature_weights, name="with Integrated Gradients weights")
        store.silhouette_with_integrated_gradients.append(silhouette_with_integrated_gradients)

        fold_logger.log_text("Calculating weighted t-SNE and silhouette for Noise Tunnel.")
        silhouette_with_noise_tunnel = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), weights=noise_tunnel_feature_weights, name="with Noise Tunnel weights")
        store.silhouette_with_noise_tunnel.append(silhouette_with_noise_tunnel)

        fold_logger.log_text("Calculating weighted t-SNE and silhouette for Deep Lift.")
        silhouette_with_deep_lift = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), weights=deep_lift_feature_weights, name="with Deep Lift weights")
        store.silhouette_with_deep_lift.append(silhouette_with_deep_lift)

        fold_logger.log_text("Calculating weighted t-SNE and silhouette for Gradient SHAP.")
        silhouette_with_gradient_shap = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), weights=gradient_shap_feature_weights, name="with Gradient SHAP weights")
        store.silhouette_with_gradient_shap.append(silhouette_with_gradient_shap)

        fold_logger.log_text("Calculating weighted t-SNE and silhouette for Feature Ablation.")
        silhouette_with_feature_ablation = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), weights=feature_ablation_feature_weights, name="with Feature Ablation weights")
        store.silhouette_with_feature_ablation.append(silhouette_with_feature_ablation)
        '''
        fold_logger.log_text("Calculating weighted t-SNE and silhouette for model with FSL.")
        silhouette_with_fsl = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), model=model_with_fsl, name="with FSL weights")
        store.silhouette_with_fsl.append(silhouette_with_fsl)
        
        fold_logger.log_text("Calculating weighted t-SNE and silhouette for model with Post-hoc FSL.")
        silhouette_with_fsl_posthoc = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), model=model_with_fsl_posthoc, name="with Post-hoc FSL weights")
        store.silhouette_with_fsl_posthoc.append(silhouette_with_fsl_posthoc)

        # Persist models

        fold_logger.log_text("Persisting models...")
        torch.save(model_without_fsl.state_dict(), f"{fold_logger.dir_path}/model_without_fsl.pt")
        torch.save(model_with_fsl.state_dict(), f"{fold_logger.dir_path}/model_with_fsl.pt")
        torch.save(model_with_fsl_posthoc.state_dict(), f"{fold_logger.dir_path}/model_with_fsl_posthoc.pt")

        fold_logger.log_text(f"Completed fold {fold_index + 1}/{num_of_tests}.")

    # End of all folds training

    logger.log_text("Completed all folds training!")
    total_training_elapsed = time.perf_counter() - total_training_start_time
    logger.log_text(f"Total training time for all folds: {total_training_elapsed:.2f} seconds.")

    # Calculate averages and standard deviations

    logger.log_text("Calculating statistics...")

    # Training time
    stat_training_time_without_weights = (pd.Series(store.training_time_without_weights).mean(), pd.Series(store.training_time_without_weights).std())
    stat_training_time_with_fsl = (pd.Series(store.training_time_with_fsl).mean(), pd.Series(store.training_time_with_fsl).std())
    stat_training_time_with_fsl_posthoc = (pd.Series(store.training_time_with_fsl_posthoc).mean(), pd.Series(store.training_time_with_fsl_posthoc).std())

    # F1 scores
    stat_f1_scores_without_weights = (pd.Series(store.f1_scores_without_weights).mean(), pd.Series(store.f1_scores_without_weights).std())
    stat_f1_scores_with_fsl = (pd.Series(store.f1_scores_with_fsl).mean(), pd.Series(store.f1_scores_with_fsl).std())
    stat_f1_scores_with_fsl_posthoc = (pd.Series(store.f1_scores_with_fsl_posthoc).mean(), pd.Series(store.f1_scores_with_fsl_posthoc).std())

    # Accuracy
    stat_accuracy_without_weights = (pd.Series(store.accuracy_without_weights).mean(), pd.Series(store.accuracy_without_weights).std())
    stat_accuracy_with_fsl = (pd.Series(store.accuracy_with_fsl).mean(), pd.Series(store.accuracy_with_fsl).std())
    stat_accuracy_with_fsl_posthoc = (pd.Series(store.accuracy_with_fsl_posthoc).mean(), pd.Series(store.accuracy_with_fsl_posthoc).std())

    # Precision
    stat_precision_without_weights = (pd.Series(store.precision_without_weights).mean(), pd.Series(store.precision_without_weights).std())
    stat_precision_with_fsl = (pd.Series(store.precision_with_fsl).mean(), pd.Series(store.precision_with_fsl).std())
    stat_precision_with_fsl_posthoc = (pd.Series(store.precision_with_fsl_posthoc).mean(), pd.Series(store.precision_with_fsl_posthoc).std())

    # Recall
    stat_recall_without_weights = (pd.Series(store.recall_without_weights).mean(), pd.Series(store.recall_without_weights).std())
    stat_recall_with_fsl = (pd.Series(store.recall_with_fsl).mean(), pd.Series(store.recall_with_fsl).std())
    stat_recall_with_fsl_posthoc = (pd.Series(store.recall_with_fsl_posthoc).mean(), pd.Series(store.recall_with_fsl_posthoc).std())

    # Silhouette
    stat_silhouette_without_weights = (pd.Series(store.silhouette_without_weights).mean(), pd.Series(store.silhouette_without_weights).std())
    stat_silhouette_integrated_gradients = (pd.Series(store.silhouette_with_integrated_gradients).mean(), pd.Series(store.silhouette_with_integrated_gradients).std())
    stat_silhouette_noise_tunnel = (pd.Series(store.silhouette_with_noise_tunnel).mean(), pd.Series(store.silhouette_with_noise_tunnel).std())
    stat_silhouette_deep_lift = (pd.Series(store.silhouette_with_deep_lift).mean(), pd.Series(store.silhouette_with_deep_lift).std())
    stat_silhouette_gradient_shap = (pd.Series(store.silhouette_with_gradient_shap).mean(), pd.Series(store.silhouette_with_gradient_shap).std())
    stat_silhouette_feature_ablation = (pd.Series(store.silhouette_with_feature_ablation).mean(), pd.Series(store.silhouette_with_feature_ablation).std())
    stat_silhouette_with_fsl = (pd.Series(store.silhouette_with_fsl).mean(), pd.Series(store.silhouette_with_fsl).std())
    stat_silhouette_with_fsl_posthoc = (pd.Series(store.silhouette_with_fsl_posthoc).mean(), pd.Series(store.silhouette_with_fsl_posthoc).std())

    # Calculate kruskal-wallis test and dunn's post-hoc test

    logger.log_text("Calculating Kruskal-Wallis and Dunn's tests...")
    training_time_kruskal_dunn_result = calculate_kruskal_dunn_3(store.training_time_without_weights, store.training_time_with_fsl, store.training_time_with_fsl_posthoc)
    f1_scores_kruskal_dunn_result = calculate_kruskal_dunn_3(store.f1_scores_without_weights, store.f1_scores_with_fsl, store.f1_scores_with_fsl_posthoc)
    accuracy_kruskal_dunn_result = calculate_kruskal_dunn_3(store.accuracy_without_weights, store.accuracy_with_fsl, store.accuracy_with_fsl_posthoc)
    precision_kruskal_dunn_result = calculate_kruskal_dunn_3(store.precision_without_weights, store.precision_with_fsl, store.precision_with_fsl_posthoc)
    recall_kruskal_dunn_result = calculate_kruskal_dunn_3(store.recall_without_weights, store.recall_with_fsl, store.recall_with_fsl_posthoc)
    silhouette_kruskal_dunn_result = calculate_kruskal_dunn_5(store.silhouette_without_weights, store.silhouette_with_integrated_gradients, store.silhouette_with_noise_tunnel, store.silhouette_with_deep_lift, store.silhouette_with_gradient_shap, store.silhouette_with_feature_ablation, store.silhouette_with_fsl, store.silhouette_with_fsl_posthoc)

    # Calculate stability metrics

    logger.log_text("Calculating stability metrics...")
    for [model_name, feature_weights_results] in [["Integrated Gradients", store.feature_weights_integrated_gradients], ["Noise Tunnel", store.feature_weights_noise_tunnel], ["Deep Lift", store.feature_weights_deep_lift], ["Gradient SHAP", store.feature_weights_gradient_shap], ["Feature Ablation", store.feature_weights_feature_ablation], ["FSL", store.feature_weights_fsl], ["Post-hoc FSL", store.feature_weights_fsl_posthoc]]:

        logger.log_text(f"Calculating statistic metrics for {model_name}...")
        results = []
        spearman_values = []
        pearson_values = []
        jaccard_values_by_k = {k: [] for k in jaccard_k_list}

        # Starts by combinations of 2

        for (_, w1), (_, w2) in combinations(enumerate(feature_weights_results), 2):
            result = {}
            spearman_result = spearman_correlation(w1, w2)
            pearson_result = pearson_correlation(w1, w2)
            result['Spearman'] = spearman_result
            result['Pearson'] = pearson_result
            spearman_values.append(spearman_result)
            pearson_values.append(pearson_result)
            for jaccard_k in jaccard_k_list:
                jaccard_result = jaccard_similarity(w1, w2, k=jaccard_k)
                result[f'Jaccard@{jaccard_k}'] = jaccard_result
                jaccard_values_by_k[jaccard_k].append(jaccard_result)
            results.append(result)

        logger.log_text(f"Stability metrics for {model_name} (Combinations of 2):")
        logger.log_dataframe(pd.DataFrame(results), filename=f"stability_combinations_{model_name}.txt")

        # Then compute mean and std

        stability_summary = {
            'Spearman Mean': np.mean(spearman_values),
            'Spearman Std': np.std(spearman_values),
            'Pearson Mean': np.mean(pearson_values),
            'Pearson Std': np.std(pearson_values)
        }
        for k, values in jaccard_values_by_k.items():
            stability_summary[f'Jaccard@{k} Mean'] = np.mean(values)
            stability_summary[f'Jaccard@{k} Std'] = np.std(values)
        logger.log_text(f"Stability metrics for {model_name} (Mean and standart deviation):")
        logger.log_dataframe(pd.DataFrame([stability_summary]), filename=f"stability_mean_and_std_{model_name}.txt")

    # Summarize and persist results

    logger.log_text("Persisting results summary...")
    summary_path = f"{logger.dir_path}/summary_results.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        messages = []
        messages.append("Traininig time without FSL: " + str(store.training_time_without_weights) + "\n statistics: " + str(stat_training_time_without_weights) + "\n")
        messages.append("Training time with FSL: " + str(store.training_time_with_fsl) + "\n statistics: " + str(stat_training_time_with_fsl) + "\n")
        messages.append("Training time with Post-hoc FSL: " + str(store.training_time_with_fsl_posthoc) + "\n statistics: " + str(stat_training_time_with_fsl_posthoc) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for training times: " + training_time_kruskal_dunn_result + "\n")
        messages.append("F1 Scores without FSL: " + str(store.f1_scores_without_weights) + "\n statistics: " + str(stat_f1_scores_without_weights) + "\n")
        messages.append("F1 Scores with FSL: " + str(store.f1_scores_with_fsl) + "\n statistics: " + str(stat_f1_scores_with_fsl) + "\n")
        messages.append("F1 Scores with Post-hoc FSL: " + str(store.f1_scores_with_fsl_posthoc) + "\n statistics: " + str(stat_f1_scores_with_fsl_posthoc) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for F1 scores: " + f1_scores_kruskal_dunn_result + "\n")
        messages.append("Accuracy without FSL: " + str(store.accuracy_without_weights) + "\n statistics: " + str(stat_accuracy_without_weights) + "\n")
        messages.append("Accuracy with FSL: " + str(store.accuracy_with_fsl) + "\n statistics: " + str(stat_accuracy_with_fsl) + "\n")
        messages.append("Accuracy with Post-hoc FSL: " + str(store.accuracy_with_fsl_posthoc) + "\n statistics: " + str(stat_accuracy_with_fsl_posthoc) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for accuracy: " + accuracy_kruskal_dunn_result + "\n")
        messages.append("Precision without FSL: " + str(store.precision_without_weights) + "\n statistics: " + str(stat_precision_without_weights) + "\n")
        messages.append("Precision with FSL: " + str(store.precision_with_fsl) + "\n statistics: " + str(stat_precision_with_fsl) + "\n")
        messages.append("Precision with Post-hoc FSL: " + str(store.precision_with_fsl_posthoc) + "\n statistics: " + str(stat_precision_with_fsl_posthoc) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for precision: " + precision_kruskal_dunn_result + "\n")
        messages.append("Recall without FSL: " + str(store.recall_without_weights) + "\n statistics: " + str(stat_recall_without_weights) + "\n")
        messages.append("Recall with FSL: " + str(store.recall_with_fsl) + "\n statistics: " + str(stat_recall_with_fsl) + "\n")
        messages.append("Recall with Post-hoc FSL: " + str(store.recall_with_fsl_posthoc) + "\n statistics: " + str(stat_recall_with_fsl_posthoc) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for recall: " + recall_kruskal_dunn_result + "\n")
        messages.append("Silhouette without FSL: " + str(store.silhouette_without_weights) + "\n statistics: " + str(stat_silhouette_without_weights) + "\n")
        messages.append("Silhouette with Integrated Gradients: " + str(store.silhouette_with_integrated_gradients) + "\n statistics: " + str(stat_silhouette_integrated_gradients) + "\n")
        messages.append("Silhouette with Noise Tunnel: " + str(store.silhouette_with_noise_tunnel) + "\n statistics: " + str(stat_silhouette_noise_tunnel) + "\n")
        messages.append("Silhouette with Deep Lift: " + str(store.silhouette_with_deep_lift) + "\n statistics: " + str(stat_silhouette_deep_lift) + "\n")
        messages.append("Silhouette with Gradient SHAP: " + str(store.silhouette_with_gradient_shap) + "\n statistics: " + str(stat_silhouette_gradient_shap) + "\n")
        messages.append("Silhouette with Feature Ablation: " + str(store.silhouette_with_feature_ablation) + "\n statistics: " + str(stat_silhouette_feature_ablation) + "\n")
        messages.append("Silhouette with FSL: " + str(store.silhouette_with_fsl) + "\n statistics: " + str(stat_silhouette_with_fsl) + "\n")
        messages.append("Silhouette with Post-hoc FSL: " + str(store.silhouette_with_fsl_posthoc) + "\n statistics: " + str(stat_silhouette_with_fsl_posthoc) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for silhouette: " + silhouette_kruskal_dunn_result + "\n")
        messages.append("PIFS with FSL: " + str(store.pifs_with_fsl) + "\n")
        messages.append("PIFS with Post-hoc FSL: " + str(store.pifs_with_fsl_posthoc) + "\n")
        messages.append("PSFI with FSL: " + str(store.psfi_with_fsl) + "\n")
        messages.append("PSFI with Post-hoc FSL: " + str(store.psfi_with_fsl_posthoc) + "\n")
        for message in messages:
            f.write(message)
            f.write("\n")

    # Generate feature position walk plots for the two models with FSL

    logger.log_text("Generating feature position walk plots...")
    '''
    generate_feature_position_walk_plot(logger, store.feature_weights_integrated_gradients, feature_columns, model_name="Integrated Gradients", informative_features=informative_features)
    generate_feature_position_walk_plot(logger, store.feature_weights_noise_tunnel, feature_columns, model_name="Noise Tunnel", informative_features=informative_features)
    generate_feature_position_walk_plot(logger, store.feature_weights_deep_lift, feature_columns, model_name="Deep Lift", informative_features=informative_features)
    generate_feature_position_walk_plot(logger, store.feature_weights_gradient_shap, feature_columns, model_name="Gradient SHAP", informative_features=informative_features)
    generate_feature_position_walk_plot(logger, store.feature_weights_feature_ablation, feature_columns, model_name="Feature Ablation", informative_features=informative_features)
    '''
    generate_feature_position_walk_plot(logger, store.feature_weights_fsl, feature_columns, model_name="FSL", informative_features=informative_features)
    generate_feature_position_walk_plot(logger, store.feature_weights_fsl_posthoc, feature_columns, model_name="Post-hoc FSL", informative_features=informative_features)
    # End of execution

    general_elapsed = time.perf_counter() - general_start_time
    logger.log_text(f"Experiment concluded for execution id: {execution_id}.")
    logger.log_text(f"Total elapsed time: {general_elapsed:.2f} seconds.")