from sklearn.preprocessing import MinMaxScaler
from modelConfigs.syntheticModules import SynthModel, SynthModelWithFSL
from trainingConfigs.multiple import multiple_training

def synth_multiple_training():
    multiple_training(
        name="synthetic",
        base_model=SynthModel,
        model_with_fsl=SynthModelWithFSL,
        weight_scaler=MinMaxScaler,
        dataset_path='data/synthetic/synth_3000samples_100features_30informative.csv',
        label_column='class',
        num_of_tests=11,
        test_percentage=0.1,
        seed=42,
        learning_rate=0.001,
        batch_size=16,
        n_epochs_base=45,
        n_epochs_fsl=45,
        n_epochs_fsl_posthoc=45,
        should_persist=True,
        num_of_informative_features_to_display=30,
        jaccard_k_list=list(range(1, 101)),
        l=0.0025
    )