from modelConfigs.liverModules import LiverModel, LiverModelWithFSL
from trainingConfigs.multiple import multiple_training

def liver_multiple_training():
    multiple_training(
        name="liver",
        base_model=LiverModel,
        model_with_fsl=LiverModelWithFSL,
        dataset_path='data/cumida/Liver_GSE22405.csv',
        label_column='type',
        ignored_columns=['samples'],
        num_of_tests=11,
        test_percentage=0.1,
        has_numeric_labels=False,
        seed=42,
        learning_rate=0.001,
        batch_size=8,
        n_epochs_base=50,
        n_epochs_fsl=500,
        n_epochs_fsl_posthoc=50,
        should_persist=True,
        num_of_informative_features_to_display=30,
        jaccard_k_list=list(range(1, 101)), 
        l=0.002
    )