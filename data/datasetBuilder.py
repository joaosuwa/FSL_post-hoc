import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd


class DatasetBuilder(Dataset):
    def __init__(self, dataframe, feature_cols, label_col, transform=None):
        self.data = dataframe
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.transform = transform
        self.features = np.empty((len(self.data),), dtype=object)
        self.np_features = np.zeros_like(self.data[feature_cols])
        self.labels = np.empty((len(self.data),), dtype=object)
        self.np_labels = np.zeros_like(self.data[label_col])
        self.__prepare_dataset__()

    def __prepare_dataset__(self):
        for idx in range(len(self.data)):
            feature, label = self.__prepare_item__(idx)
            self.features[idx] = feature
            self.np_features[idx] = feature.numpy()
            self.labels[idx] = label
            self.np_labels[idx] = label.numpy()

    def __prepare_item__(self, idx):
        features_df = self.data.loc[idx, self.feature_cols]
        # Convert all feature columns to numeric, so that it may transform it to a PyTorch Tensor
        features = torch.tensor(features_df.apply(pd.to_numeric, errors='coerce').values, dtype=torch.float32)
        label = torch.tensor(self.data.loc[idx, self.label_col], dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]