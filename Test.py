import torch
from torch import nn
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

# Classes

class FeatureSelectionV2(nn.Module):
    def __init__(self, in_features, internal_model, freeze_internal_model=True):
        super().__init__()
        initial_weight = 0.1 * torch.ones(1, in_features).to(device)
        self.weight = nn.Parameter(initial_weight)
        self.activation = nn.ReLU()
        self.internal_model = internal_model
        for param in self.internal_model.parameters():
            param.requires_grad = not freeze_internal_model

    def forward(self, input):
        weighted_input = input * self.get_activated_weights()
        output, M_loss = self.internal_model.forward(weighted_input)
        return output, M_loss

    def get_weights(self):
        return self.weight

    def get_activated_weights(self):
        return self.activation(self.weight)
    
    def train(self):
        super().train()
        self.internal_model.train()


# Dummy data

# Generate random features: shape [100, 10]
X_train = np.random.rand(100, 10)
# Generate random binary labels: shape [100]
y_train = np.random.randint(0, 2, size=100)

# Initialize TabNet
model = TabNetClassifier()
model.fit(X_train, y_train)

layer_count = sum(1 for module in model.network.modules() if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential))
print(f"Number of layers: {layer_count}")

total_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")

fs_model = FeatureSelectionV2(10, model.network)

# Optimizer and loss
optimizer = torch.optim.Adam(fs_model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

initial_params = {name: param.clone() for name, param in fs_model.internal_model.named_parameters()}

# Training loop
for epoch in range(30):
    fs_model.train()
    optimizer.zero_grad()
    y_pred, M_loss = fs_model(torch.from_numpy(X_train).float().to(device))
    loss = criterion(y_pred, torch.from_numpy(y_train).long().to(device))
    loss.backward()
    for name, param in fs_model.internal_model.named_parameters():
        if param.grad is not None:
            print(f"{name} gradient norm: {param.grad.norm().item():.4f}")
        else:
            print(f"{name} has no gradient.")
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

for name, param in fs_model.internal_model.named_parameters():
    if not torch.equal(param, initial_params[name]):
        print(f"{name} has changed.")
    else:
        print(f"{name} is unchanged.")

print(fs_model.get_activated_weights())
print(model.feature_importances_)