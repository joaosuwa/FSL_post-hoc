from modelConfigs.spamModules import SpamModel, SpamModelWithFSL
from trainingConfigs.multiple import multiple_training

def spam_multiple_training():
    multiple_training(
        name='spam',
        base_model=SpamModel,
        model_with_fsl=SpamModelWithFSL,
        dataset_path='data/spam/spam-dataset_normalized.csv',
        label_column='Prediction',
        has_numeric_labels=True,
        ignored_columns=[],
        num_of_tests=3,
        test_percentage=0.3,
        seed=None,
        learning_rate=0.001,
        batch_size=16,
        n_epochs_base=40,
        n_epochs_fsl=40,
        n_epochs_fsl_posthoc=40,
        should_persist=True,
        num_of_informative_features_to_display=30,
        jaccard_k_list=list(range(1, 101)),
        l=0.001
    )