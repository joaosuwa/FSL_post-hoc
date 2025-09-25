from torch import nn
from featureSelectionLayer import FeatureSelectionV2

class LeukemiaModel(nn.Module):
        def __init__(self, name: str = ''):
          super().__init__()
          self.name=name
          self.block_1=nn.Sequential(
              nn.Linear(22283, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(100, 200),
              nn.BatchNorm1d(200),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(200, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(100, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Linear(100, 1)
          )
        def forward(self, x):
          return self.block_1(x)

class LeukemiaModelWithFSL(nn.Module):
        def __init__(self, name: str = ''):
          super().__init__()
          self.name = name
          self.block_1=nn.Sequential(
              FeatureSelectionV2(22283),
              nn.Linear(22283, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(100, 200),
              nn.BatchNorm1d(200),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(200, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Dropout(0.2),
              nn.Linear(100, 100),
              nn.BatchNorm1d(100),
              nn.ReLU(),
              nn.Linear(100, 1)
          )
        def forward(self, x):
          return self.block_1(x)