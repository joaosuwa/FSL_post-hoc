import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report


def calculate_prediction_metrics(logger, model, test_dataloader, print_function=print, name=None):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_dataloader:
            y_pred_logits = model(X_batch)
            y_pred_prob = torch.sigmoid(y_pred_logits) 
            y_pred_label = torch.round(y_pred_prob) 
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(y_pred_label.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    report = classification_report(y_true, y_pred)
    print_function(f"Validation Metrics for model_without_fsl:")
    print_function(f"F1 Score: {f1:.4f}")
    print_function(f"Accuracy: {acc:.4f}")
    print_function(f"Precision: {precision:.4f}")
    print_function(f"Recall: {recall:.4f}")
    print_function(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{logger.dir_path}/confusion_matrix{f"_({name})" if name else ""}.png')
    plt.close()

    return f1, acc, precision, recall