import torch
from pytorch_tabnet.tab_model import TabNetClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"


class TabModelWrapper():
    def __init__(self, name: str = ''):
        super().__init__()
        self.tab_model = TabNetClassifier()
        self.name = name
        self.block_1 = [self]
        
    def forward(self, x):
        return self.tab_model.network.forward(x)
    
    def parameters(self):
        return self.tab_model.network.parameters()
    
    def fit(self, x, y):
        self.tab_model.fit(x.values, y.values)

    def get_weights(self):
        return torch.from_numpy(self.tab_model.feature_importances_).to(device)
    
    def get_activated_weights(self):
        return self.get_weights()
    
    def train(self, value=True):
        self.tab_model.network.train(value)