import pandas as pd
from data.loadDataset import loadDataset
from modelConfigs.liverModules_14520_U133A import LiverModel, LiverModelWithFSL
from trainingTestStep import trainingModule
from utils import displayTopFeatures
from featureSelectionLayer import freezeParams, transfer_weights
from wtsne import WTSNE
from sklearn.preprocessing import LabelEncoder

def liver_14520_U133A_Training():

    df = pd.read_csv('data/cumida/Liver_GSE14520_U133A.csv')
    le = LabelEncoder()
    df['type'] = le.fit_transform(df['type']) # Now all the cancer types are label as 0, and the normal as 1

    feature_cols = df.columns[2:].to_list()
    labelCollumnStr = 'type'

    Liver_dataset, train_dataloader, test_dataloader = loadDataset(df=df, 
                                                                    feature_cols=feature_cols, 
                                                                    labelCollumn=labelCollumnStr, 
                                                                    batch_size=32,
                                                                    percentageSplit=0.8)

    modelLiver = LiverModel()
    print("Training model without FSL \n")
    modelLiver = trainingModule(modelLiver, train_dataloader, test_dataloader, n_epochs=30, earlyStop=True, isFSLpresent=False)

    modelLiverFSL = LiverModelWithFSL(name="Model with embedded FSL")
    print("Training model with FSL \n")

    modelLiverFSL = trainingModule(modelLiverFSL, train_dataloader, test_dataloader, n_epochs=30, earlyStop=True, isFSLpresent=True)
    displayTopFeatures(modelLiverFSL, feature_cols)

    modelLiverFSL_PostHoc = LiverModelWithFSL(name="Model with post-hoc FSL")
    transfer_weights(modelLiver, modelLiverFSL_PostHoc)
    freezeParams(modelLiverFSL_PostHoc)

    print("Training model with FSL (post-hoc)\n")

    modelLiverFSL_PostHoc = trainingModule(modelLiverFSL_PostHoc, train_dataloader, test_dataloader, n_epochs=30, earlyStop=False, isFSLpresent=True)
    displayTopFeatures(modelLiverFSL_PostHoc, feature_cols)

    datasetName='Liver'

    WTSNE(Liver_dataset, datasetName=datasetName)
    WTSNE(Liver_dataset, datasetName=datasetName, model=modelLiverFSL)
    WTSNE(Liver_dataset, datasetName=datasetName, model=modelLiverFSL_PostHoc)