import time
import numpy as np
import torch
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from feature_position_walk import generate_feature_position_walk_plot
from trainingConfigs.execution_store import TabNetExecutionStore
from trainingTestStep import trainingModule
from utils import LogPrinter, calculate_kruskal_dunn_2, generate_execution_id, displayTopFeatures, find_normalized_weights, get_feature_rankings, get_feature_weights_as_numpy, persist_wtsne_input
from data.loadDataset import folds_to_dataloaders, numpy_to_dataloaders
from metrics import Selection_Accuracy, jaccard_similarity, pearson_correlation, spearman_correlation
from wtsne import WTSNEv2
from evaluation import calculate_prediction_metrics
from featureSelectionLayer import TabNetFeatureSelectionV2
from tabModelWrapper import TabModelWrapper

def xor_tabnet_multiple_training():
    tabnet_multiple_training(
        name="xor_tabnet",
        dataset_path='data/xor/xor_500samples_50features.csv',
        label_column='class',
        num_of_tests=11,
        test_percentage=0.2,
        seed=None,
        learning_rate=0.002,
        batch_size=32,
        n_epochs_fsl=100,
        should_persist=True,
        num_of_informative_features_to_display=10,
        jaccard_k_list=list(range(1, 51)),
        l=0.0025
    )

def tabnet_multiple_training(dataset_path, label_column, name='tabnet', has_numeric_labels=True, ignored_columns=[], num_of_tests=3, test_percentage=0.1, seed=None, batch_size=32, n_epochs_fsl=50,
                             learning_rate=0.01, should_persist=True, num_of_informative_features_to_display=10, jaccard_k_list=None, l=0.001, scaler=StandardScaler):
    general_start_time = time.perf_counter()

    # TabNet model

    TabModel = TabModelWrapper
    FSLTabModel = TabNetFeatureSelectionV2
    
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

    test_dataloader = numpy_to_dataloaders(X_test, y_test, batch_size=batch_size)

    # Create KFolds

    logger.log_text("Creating folds...")
    skf = StratifiedKFold(n_splits=num_of_tests, shuffle=True, random_state=seed)
    folds = list(skf.split(X_train, y_train))
    logger.log_text(f"Number of folds created: {len(folds)}.")

    # Create store to persist fold results

    store = TabNetExecutionStore()

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
        X_fold_train_dataframe = pd.DataFrame(scaler().fit_transform(X_fold_train), columns=X.columns)
        X_fold_val_dataframe = pd.DataFrame(scaler().fit_transform(X_fold_val), columns=X.columns)
        train_dataloader, val_dataloader = folds_to_dataloaders(X_fold_train_dataframe, y_fold_train, X_fold_val_dataframe, y_fold_val, batch_size=batch_size)
        fold_logger.log_text(f"Fold {fold_index + 1} - Train samples: {X_fold_train_dataframe.shape[0]}.")
        fold_logger.log_text(f"Fold {fold_index + 1} - Validation samples: {X_fold_val_dataframe.shape[0]}.")

        # Train three models: without FSL, with FSL, with FSL post-hoc and persist the execution times

        fold_logger.log_text("Training model without FSL.")
        tab_model = TabModel(name='TabModel')
        tab_model.fit(X_fold_train, y_fold_train)
        start_time = time.perf_counter()
        elapsed = time.perf_counter() - start_time
        store.training_time_tabnet.append(elapsed)
        
        fold_logger.log_text("Training model with FSL.")
        model_with_fsl = FSLTabModel(len(feature_columns), tab_model, name='FSLModel')
        start_time = time.perf_counter()
        model_with_fsl = trainingModule(model_with_fsl, train_dataloader, val_dataloader, n_epochs=n_epochs_fsl, 
                                        earlyStop=False, isFSLpresent=True, seed=seed, print_function=fold_logger.log_text, lr=learning_rate, l=l, is_multiclass=True, num_classes=len(feature_columns))
        elapsed = time.perf_counter() - start_time
        store.training_time_with_fsl.append(elapsed)

        # Display top features for the two models with FSL

        displayTopFeatures(tab_model, feature_columns, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-tab-model.txt"))
        displayTopFeatures(model_with_fsl, feature_columns, print_function=fold_logger.log_text, display_function=lambda df: fold_logger.log_dataframe(df, filename="top-features-fsl.txt"))

        # Persist weights as original wtsne input

        persist_wtsne_input(tab_model, feature_columns, name="tab", logger=fold_logger)
        persist_wtsne_input(model_with_fsl, feature_columns, name="fsl", logger=fold_logger)

        # Get feature weights for the two models with FSL

        weights_with_tabmodel = get_feature_weights_as_numpy(tab_model)
        store.feature_weights_results_tabnet.append(weights_with_tabmodel)
        weights_with_fsl = get_feature_weights_as_numpy(model_with_fsl)
        store.feature_weights_results_with_fsl.append(weights_with_fsl)

        # Persist feature weights for the two models with FSL

        fold_logger.log_text("Persisting feature weights...")
        fold_logger.log_np_array(tab_model, filename=f"tabnet_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(weights_with_fsl, filename=f"fsl_feature_weights.txt", fmt='%f')
    
        # Get normalized weights for the two models with FSL

        tabnet_normalized_feature_weights = find_normalized_weights(tab_model)
        fsl_normalized_feature_weights = find_normalized_weights(model_with_fsl)

        # Persist normalized feature weights for the two models with FSL

        fold_logger.log_text("Persisting normalized feature weights...")
        fold_logger.log_np_array(tabnet_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"tabnet_normalized_feature_weights.txt", fmt='%f')
        fold_logger.log_np_array(fsl_normalized_feature_weights.squeeze().detach().cpu().numpy(), filename=f"fsl_normalized_feature_weights.txt", fmt='%f')

        # Persist feature rankings for the two models with FSL

        fold_logger.log_text("Persisting feature rankings...")
        fold_logger.log_np_array(get_feature_rankings(tab_model, feature_columns), filename=f"tabnet_feature_rankings.txt", fmt='%s')
        fold_logger.log_np_array(get_feature_rankings(model_with_fsl, feature_columns), filename=f"fsl_feature_rankings.txt", fmt='%s')

        # Calculate feature selection metrics for the two models with FSL

        if len(informative_features) > 0:
            fold_logger.log_text("Calculating feature selection metrics...")
            pifs, psfi = Selection_Accuracy(tabnet_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_tabnet.append(pifs)
            store.psfi_tabnet.append(psfi)
            pifs, psfi = Selection_Accuracy(fsl_normalized_feature_weights.squeeze().detach().cpu().clone(), informative_features, 
                               num_of_informative_features_to_display, feature_columns, print_function=fold_logger.log_text)
            store.pifs_with_fsl.append(pifs)
            store.psfi_with_fsl.append(psfi)
        else:
            fold_logger.log_text("No informative features found, skipping feature selection metrics.")

        # Calculate prediction metrics for the three models

        fold_logger.log_text("Calculating prediction metrics for TabNet.")
        f1, acc, precision, recall = calculate_prediction_metrics(fold_logger, tab_model, test_dataloader, print_function=fold_logger.log_text, name="Tabnet")
        store.f1_scores_tabnet.append(f1)
        store.accuracy_tabnet.append(acc)
        store.precision_tabnet.append(precision)
        store.recall_tabnet.append(recall)

        fold_logger.log_text("Calculating prediction metrics for model with FSL.")
        f1, acc, precision, recall = calculate_prediction_metrics(fold_logger, model_with_fsl, test_dataloader, print_function=fold_logger.log_text, name="with FSL")
        store.f1_scores_with_fsl.append(f1)
        store.accuracy_with_fsl.append(acc)
        store.precision_with_fsl.append(precision)
        store.recall_with_fsl.append(recall)

        # Calculate weighted t-SNE and silhouette for the three models

        fold_logger.log_text("Calculating standart t-SNE and silhouette without weights.")
        silhouette_tabnet = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), name="without weights")
        store.silhouette_tabnet.append(silhouette_tabnet)
        
        fold_logger.log_text("Calculating weighted t-SNE and silhouette for TabNet.")
        silhouette_with_fsl_posthoc = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), model=tab_model, name="with Post-hoc FSL weights")
        store.silhouette_tabnet.append(silhouette_with_fsl_posthoc)

        fold_logger.log_text("Calculating weighted t-SNE and silhouette for model with FSL.")
        silhouette_with_fsl = WTSNEv2(fold_logger, X.to_numpy(), y.to_numpy(), model=model_with_fsl, name="with FSL weights")
        store.silhouette_with_fsl.append(silhouette_with_fsl)
        
        # Persist models

        fold_logger.log_text("Persisting models...")
        # TODO: Persist TabNet
        torch.save(model_with_fsl.state_dict(), f"{fold_logger.dir_path}/model_with_fsl.pt")

        fold_logger.log_text(f"Completed fold {fold_index + 1}/{num_of_tests}.")

    # End of all folds training

    logger.log_text("Completed all folds training!")
    total_training_elapsed = time.perf_counter() - total_training_start_time
    logger.log_text(f"Total training time for all folds: {total_training_elapsed:.2f} seconds.")

    # Calculate averages and standard deviations

    logger.log_text("Calculating statistics...")
    stat_training_time_tabnet = (pd.Series(store.training_time_tabnet).mean(), pd.Series(store.training_time_tabnet).std())
    stat_training_time_with_fsl = (pd.Series(store.training_time_with_fsl).mean(), pd.Series(store.training_time_with_fsl).std())
    stat_f1_scores_tabnet = (pd.Series(store.f1_scores_tabnet).mean(), pd.Series(store.f1_scores_tabnet).std())
    stat_f1_scores_with_fsl = (pd.Series(store.f1_scores_with_fsl).mean(), pd.Series(store.f1_scores_with_fsl).std())
    stat_accuracy_tabnet = (pd.Series(store.accuracy_tabnet).mean(), pd.Series(store.accuracy_tabnet).std())
    stat_accuracy_with_fsl = (pd.Series(store.accuracy_with_fsl).mean(), pd.Series(store.accuracy_with_fsl).std())
    stat_precision_tabnet = (pd.Series(store.precision_tabnet).mean(), pd.Series(store.precision_tabnet).std())
    stat_precision_with_fsl = (pd.Series(store.precision_with_fsl).mean(), pd.Series(store.precision_with_fsl).std())
    stat_recall_tabnet = (pd.Series(store.recall_tabnet).mean(), pd.Series(store.recall_tabnet).std())
    stat_recall_with_fsl = (pd.Series(store.recall_with_fsl).mean(), pd.Series(store.recall_with_fsl).std())
    stat_silhouette_tabnet = (pd.Series(store.silhouette_tabnet).mean(), pd.Series(store.silhouette_tabnet).std())
    stat_silhouette_with_fsl = (pd.Series(store.silhouette_with_fsl).mean(), pd.Series(store.silhouette_with_fsl).std())

    # Calculate kruskal-wallis test and dunn's post-hoc test

    logger.log_text("Calculating Kruskal-Wallis and Dunn's tests...")
    training_time_kruskal_dunn_result = calculate_kruskal_dunn_2(store.training_time_tabnet, store.training_time_with_fsl)
    f1_scores_kruskal_dunn_result = calculate_kruskal_dunn_2(store.f1_scores_tabnet, store.f1_scores_with_fsl)
    accuracy_kruskal_dunn_result = calculate_kruskal_dunn_2(store.accuracy_tabnet, store.accuracy_with_fsl)
    precision_kruskal_dunn_result = calculate_kruskal_dunn_2(store.precision_tabnet, store.precision_with_fsl)
    recall_kruskal_dunn_result = calculate_kruskal_dunn_2(store.recall_tabnet, store.recall_with_fsl)
    silhouette_kruskal_dunn_result = calculate_kruskal_dunn_2(store.silhouette_tabnet, store.silhouette_with_fsl)

    # Calculate stability metrics

    logger.log_text("Calculating stability metrics...")
    for [model_name, feature_weights_results] in [["TabNet", store.feature_weights_results_tabnet], ["FSL", store.feature_weights_results_with_fsl]]:

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
        messages.append("Traininig time TabNet: " + str(store.training_time_tabnet) + "\n statistics: " + str(stat_training_time_tabnet) + "\n")
        messages.append("Training time with FSL: " + str(store.training_time_with_fsl) + "\n statistics: " + str(stat_training_time_with_fsl) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for training times: " + training_time_kruskal_dunn_result + "\n")
        messages.append("F1 Scores TabNet: " + str(store.f1_scores_tabnet) + "\n statistics: " + str(stat_f1_scores_tabnet) + "\n")
        messages.append("F1 Scores with FSL: " + str(store.f1_scores_with_fsl) + "\n statistics: " + str(stat_f1_scores_with_fsl) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for F1 scores: " + f1_scores_kruskal_dunn_result + "\n")
        messages.append("Accuracy TabNet: " + str(store.accuracy_tabnet) + "\n statistics: " + str(stat_accuracy_tabnet) + "\n")
        messages.append("Accuracy with FSL: " + str(store.accuracy_with_fsl) + "\n statistics: " + str(stat_accuracy_with_fsl) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for accuracy: " + accuracy_kruskal_dunn_result + "\n")
        messages.append("Precision TabNet: " + str(store.precision_tabnet) + "\n statistics: " + str(stat_precision_tabnet) + "\n")
        messages.append("Precision with FSL: " + str(store.precision_with_fsl) + "\n statistics: " + str(stat_precision_with_fsl) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for precision: " + precision_kruskal_dunn_result + "\n")
        messages.append("Recall TabNet: " + str(store.recall_tabnet) + "\n statistics: " + str(stat_recall_tabnet) + "\n")
        messages.append("Recall with FSL: " + str(store.recall_with_fsl) + "\n statistics: " + str(stat_recall_with_fsl) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for recall: " + recall_kruskal_dunn_result + "\n")
        messages.append("Silhouette TabNet: " + str(store.silhouette_tabnet) + "\n statistics: " + str(stat_silhouette_tabnet) + "\n")
        messages.append("Silhouette with FSL: " + str(store.silhouette_with_fsl) + "\n statistics: " + str(stat_silhouette_with_fsl) + "\n")
        messages.append("Kruskal-Wallis and Dunn's test for silhouette: " + silhouette_kruskal_dunn_result + "\n")
        messages.append("PIFS with FSL: " + str(store.pifs_with_fsl) + "\n")
        messages.append("PIFS with TabNet: " + str(store.pifs_tabnet) + "\n")
        messages.append("PSFI with FSL: " + str(store.psfi_with_fsl) + "\n")
        messages.append("PSFI with TabNet: " + str(store.psfi_tabnet) + "\n")
        for message in messages:
            f.write(message)
            f.write("\n")

    # Generate feature position walk plots for the two models with FSL

    logger.log_text("Generating feature position walk plots...")
    generate_feature_position_walk_plot(logger, store.feature_weights_results_tabnet, feature_columns, model_name="TabNet", informative_features=informative_features)
    generate_feature_position_walk_plot(logger, store.feature_weights_results_with_fsl, feature_columns, model_name="FSL", informative_features=informative_features)

    # End of execution

    general_elapsed = time.perf_counter() - general_start_time
    logger.log_text(f"Experiment concluded for execution id: {execution_id}.")
    logger.log_text(f"Total elapsed time: {general_elapsed:.2f} seconds.")