from data.datasetBuilder import DatasetBuilder
import torch
from torch.utils.data import random_split


def loadDataset(df, feature_cols, labelCollumn: str, batch_size=32, percentageSplit=0.8):
    dataset = DatasetBuilder(df, feature_cols, labelCollumn)

    len_training_data = int(percentageSplit * len(dataset))
    len_testint_data = len(dataset) - len_training_data

    generator1 = torch.Generator().manual_seed(42)

    training_data, test_data = random_split(dataset, [len_training_data, len_testint_data],generator=generator1)
    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return dataset, train_dataloader, test_dataloader