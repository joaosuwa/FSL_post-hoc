from torch import nn
from featureSelectionLayer import FeatureSelectionV2

class XorModel(nn.Module):
    def __init__(self, name:str = ''):
        super().__init__()
        self.name = name
        self.block_1=nn.Sequential(
            nn.Linear(50, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.block_1(x)

class XorModelWithFSL(nn.Module):
    def __init__(self, name:str = ''):
        super().__init__()
        self.name = name
        self.block_1=nn.Sequential(
            FeatureSelectionV2(50),
            nn.Linear(50, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.block_1(x)
