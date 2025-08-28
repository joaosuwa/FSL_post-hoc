import torch
import pandas as pd
from data.loadDataset import loadDataset
from modelConfigs.xorModules import XorModel, XorModelWithFSL
from trainingTestStep import trainingModule
from utils import displayTopFeatures, find_normalized_weights
from metrics import Selection_Accuracy
from featureSelectionLayer import freezeParams, transfer_weights
from wtsne import WTSNE

def xorTraining():

    df = pd.read_csv('data/xor/xor_500samples_50features.csv')
    feature_cols = df.columns[:-1].to_list()
    labelCollumnStr = 'class'

    Xor_dataset, train_dataloader, test_dataloader = loadDataset(df=df, 
                                                                    feature_cols=feature_cols, 
                                                                    labelCollumn=labelCollumnStr, 
                                                                    batch_size=32,
                                                                    percentageSplit=0.8)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    modelXor = XorModel()
    print("Training model without FSL \n")
    modelXor = trainingModule(modelXor, train_dataloader, test_dataloader, n_epochs=10, earlyStop=False, isFSLpresent=False)

    modelXorFSL = XorModelWithFSL(name="Model with embedded FSL")
    print("Training model with FSL \n")
    modelXorFSL = trainingModule(modelXorFSL, train_dataloader, test_dataloader, n_epochs=10, earlyStop=False, isFSLpresent=True)

    displayTopFeatures(modelXorFSL, feature_cols)
    informative_features = [col for col in df.columns if 'informative' in col]
    normalized_weights_feature_selection = find_normalized_weights(modelXorFSL)
    Selection_Accuracy(normalized_weights_feature_selection.squeeze().detach().cpu(), informative_features, 5, feature_cols)

    modelXorFSL_PostHoc = XorModelWithFSL(name="Model with post-hoc FSL")
    transfer_weights(modelXor, modelXorFSL_PostHoc)
    freezeParams(modelXorFSL_PostHoc)
    print("Training model with FSL (post-hoc)\n")

    modelXorFSL_PostHoc = trainingModule(modelXorFSL_PostHoc, train_dataloader, test_dataloader, n_epochs=10, earlyStop=False, isFSLpresent=True)
    displayTopFeatures(modelXorFSL_PostHoc, feature_cols)
    informative_features = [col for col in df.columns if 'informative' in col]
    normalized_weights_feature_selection = find_normalized_weights(modelXorFSL_PostHoc)
    Selection_Accuracy(normalized_weights_feature_selection.squeeze().detach().cpu(), informative_features, 5, feature_cols)

    datasetName='Xor'

    WTSNE(Xor_dataset, datasetName=datasetName)
    WTSNE(Xor_dataset, datasetName=datasetName, model=modelXorFSL)
    WTSNE(Xor_dataset, datasetName=datasetName, model=modelXorFSL_PostHoc)