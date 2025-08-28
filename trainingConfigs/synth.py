import torch
import pandas as pd
from data.loadDataset import loadDataset
from modelConfigs.syntheticModules import SynthModel, SynthModelWithFSL
from trainingTestStep import trainingModule
from utils import displayTopFeatures, find_normalized_weights
from metrics import Selection_Accuracy
from featureSelectionLayer import freezeParams, transfer_weights
from wtsne import WTSNE
from PIFS_PSFI import PIFS_PSFI

def synthTraining():

    df = pd.read_csv('data/synthetic/synth_3000samples_100features_30informative.csv')
    feature_cols = df.columns[:-1].to_list()
    labelCollumnStr = 'class'
    informative_features = [col for col in df.columns if 'informative' in col]

    Synthetic_dataset, train_dataloader, test_dataloader = loadDataset(df=df, 
                                                                    feature_cols=feature_cols, 
                                                                    labelCollumn=labelCollumnStr, 
                                                                    batch_size=32,
                                                                    percentageSplit=0.8)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Model without the FSL
    modelSynth = SynthModel()
    print("Training model without FSL \n")
    modelSynth = trainingModule(modelSynth, train_dataloader, test_dataloader, n_epochs=40, earlyStop=True, isFSLpresent=False)

    # Model with FSL being trained without freezing the neural network
    modelSynthFSL = SynthModelWithFSL(name="Model with embedded FSL")
    print("Training model with FSL \n")
    modelSynthFSL = trainingModule(modelSynthFSL, train_dataloader, test_dataloader, n_epochs=40, earlyStop=True, isFSLpresent=True)

    # Displaying the 30 most informative features and calculating the PIFS and PSFI
    displayTopFeatures(modelSynthFSL, feature_cols)
    normalized_weights_feature_selection = find_normalized_weights(modelSynthFSL)
    Selection_Accuracy(normalized_weights_feature_selection.squeeze().detach().cpu(), informative_features, 30, feature_cols)

    # Model with FSL but freezing the neural network
    modelSynthFSL_PostHoc = SynthModelWithFSL(name="Model with post-hoc FSL")
    transfer_weights(modelSynth, modelSynthFSL_PostHoc)
    freezeParams(modelSynthFSL_PostHoc)

    print("Training model with FSL (post-hoc)\n")
    modelSynthFSL_PostHoc = trainingModule(modelSynthFSL_PostHoc, train_dataloader, test_dataloader, n_epochs=15, earlyStop=False, isFSLpresent=True)

    displayTopFeatures(modelSynthFSL_PostHoc, feature_cols)
    normalized_weights_feature_selection = find_normalized_weights(modelSynthFSL_PostHoc)
    Selection_Accuracy(normalized_weights_feature_selection.squeeze().detach().cpu(), informative_features, 30, feature_cols)

    models = [modelSynthFSL, modelSynthFSL_PostHoc]


    datasetName='Synth'

    PIFS_PSFI(models, feature_cols, datasetName)

    WTSNE(Synthetic_dataset, datasetName=datasetName)
    WTSNE(Synthetic_dataset, datasetName=datasetName, model=modelSynthFSL)
    WTSNE(Synthetic_dataset, datasetName=datasetName, model=modelSynthFSL_PostHoc)
