from data.datasetBuilder import DatasetBuilder
import torch
from torch.utils.data import random_split, TensorDataset, DataLoader


def loadDataset(df, feature_cols, labelCollumn: str, batch_size=32, percentageSplit=0.8):
    dataset = DatasetBuilder(df, feature_cols, labelCollumn)

    len_training_data = int(percentageSplit * len(dataset))
    len_testint_data = len(dataset) - len_training_data

    generator1 = torch.Generator().manual_seed(42)

    training_data, test_data = random_split(dataset, [len_training_data, len_testint_data],generator=generator1)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return dataset, train_dataloader, test_dataloader


def numpy_to_dataloaders(X, y, batch_size=32, shuffle=True):
    print(y)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def folds_to_dataloaders(X_train, y_train, X_val, y_val, batch_size=32, shuffle=True):
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader
