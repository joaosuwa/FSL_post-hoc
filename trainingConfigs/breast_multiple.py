from modelConfigs.BreastModules import BreastModel, BreastModelModelWithFSL
from trainingConfigs.multiple import multiple_training

def breast_multiple_training():
    multiple_training(
        name="breast",
        base_model=BreastModel,
        model_with_fsl=BreastModelModelWithFSL,
        dataset_path='data/cumida/Breast_GSE45827.csv',
        is_multiclass=True,
        num_classes=6,
        label_column='type',
        ignored_columns=['samples'],
        num_of_tests=15,
        test_percentage=0.1,
        has_numeric_labels=False,
        seed=42,
        learning_rate=0.001,
        batch_size=8,
        n_epochs_base=250,
        n_epochs_fsl=250,
        n_epochs_fsl_posthoc=250,
        should_persist=True,
        num_of_informative_features_to_display=30,
        jaccard_k_list=list(range(1, 101)), 
        l=0.0025
    )