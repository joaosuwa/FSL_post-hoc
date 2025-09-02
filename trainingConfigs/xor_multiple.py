from modelConfigs.xorModules import XorModel, XorModelWithFSL
from trainingConfigs.multiple import multiple_training

def xor_multiple_training():
    multiple_training(
        name="xor",
        base_model=XorModel,
        model_with_fsl=XorModelWithFSL,
        dataset_path='data/xor/xor_500samples_50features.csv',
        label_column='class',
        num_of_tests=11,
        test_percentage=0.2,
        seed=None,
        learning_rate=0.01,
        batch_size=32,
        n_epochs_base=50,
        n_epochs_fsl=50,
        n_epochs_fsl_posthoc=50,
        should_persist=True,
        num_of_informative_features_to_display=10,
        jaccard_k_list=list(range(1, 51))
    )