import numpy as np
from sklearn.utils import compute_class_weight
import torch
from torch import nn
from metrics import accuracy_fn
import copy
from tqdm.auto import tqdm
from featureSelectionLayer import fs_layer_regularization
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               regularization=None,
               print_function=print,
               l=0.001):
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for _, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass        
        y_pred_logits = model(X)
        y_pred_prob = torch.sigmoid(y_pred_logits)  # Apply sigmoid for probabilities
        y_pred_label = torch.round(y_pred_prob)  # Apply round for predicted labels
        #y_pred_label = torch.argmax(y_pred_logits, dim=1)  

        # 2. Calculate loss
        # Ensure y is the same dtype as y_pred_logits for BCEWithLogitsLoss
        #loss = loss_fn(y_pred_logits, y)
        loss = loss_fn(y_pred_logits, y.unsqueeze(1).float())

        if regularization is not None:
            reg = regularization(model, l)
            loss += reg

        train_loss += loss

        # Calculate accuracy based on predicted labels
        #train_acc += accuracy_fn(y_true=y, y_pred=y_pred_logits.argmax(dim=1))
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred_label.squeeze(1))

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print_function(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              print_function=print):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode():
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)
            test_pred_prob = torch.sigmoid(test_pred_logits)  # Apply sigmoid for probabilities
            test_pred_label = torch.round(test_pred_prob)  # Apply round for predicted labels

            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred_logits, y.unsqueeze(1).float())
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred_label.squeeze(1) # Use predicted labels for accuracy
            )

        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print_function(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    return test_acc, test_loss

def trainingModule(model, train_dataloader, validation_dataloader, n_epochs, earlyStop = False, isFSLpresent = False, print_function=print, seed=42, lr=0.001, l=0.001, patience=100):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    weightDecayValue = 0.001

    if (isFSLpresent):
        regularization = fs_layer_regularization
    else:
        regularization = None

    all_labels = []

    for batch in train_dataloader:
        _, labels = batch  # assuming your dataset returns (input, label)
        all_labels.extend(labels.tolist())  # convert tensor to list and add to all_labels

    loss_fn = get_loss_function(train_dataloader, print_function=print_function)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weightDecayValue)

    epochs = n_epochs

    if(earlyStop):
        best_test_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

    for epoch in tqdm(range(epochs)):
        print_function(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            regularization=regularization,
            print_function=print_function,
            l=l
        )

        _, test_loss = test_step(data_loader=validation_dataloader,

            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            print_function=print_function
        )
        if (earlyStop):
            if test_loss <= best_test_loss:
                best_test_loss = test_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print_function(f"Early stopping at epoch {epoch} as test accuracy did not improve for {patience} epochs.")
                    break

    if (earlyStop):
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print_function("Loaded best model state based on test accuracy.")

    return model

def get_loss_function(train_dataloader, print_function=print):
    # Initialize counters
    positive_count = 0
    negative_count = 0

    # Loop through the dataloader
    for batch in train_dataloader:
        _, labels = batch 
        labels = labels.view(-1)
        print(labels)
        positive_count += (labels == 1).sum().item()
        negative_count += (labels == 0).sum().item()

    # Avoid division by zero
    epsilon = 1e-8

    # Calculate pos weight
    pos_weight_value = negative_count / (positive_count + epsilon)

    # Convert to tensor
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    # Generate loss function
    print_function(f"Generating loss function with pos weight of: {pos_weight}...")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return loss_fn
