from sklearn.preprocessing import MinMaxScaler
from modelConfigs.leukemiaModules import LeukemiaModel, LeukemiaModelWithFSL
from trainingConfigs.multiple import multiple_training

def leukemia_multiple_training():
    multiple_training(
        name="leukemia",
        base_model=LeukemiaModel,
        model_with_fsl=LeukemiaModelWithFSL,
        dataset_path='data/cumida/Leukemia_GSE22529_U133A.csv',
        weight_scaler=MinMaxScaler,
        is_multiclass=False,
        label_column='type',
        ignored_columns=['samples'],
        num_of_tests=11,
        test_percentage=0.15,
        has_numeric_labels=False,
        seed=42,
        learning_rate=0.0005,
        batch_size=4,
        n_epochs_base=100,
        n_epochs_fsl=400,
        n_epochs_fsl_posthoc=800,
        should_persist=True,
        num_of_informative_features_to_display=30,
        jaccard_k_list=list(range(1, 101)), 
        l=0.0025
    )