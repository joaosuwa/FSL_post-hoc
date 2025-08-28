from utils import min_max_normalized_weights
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from metrics import silhouetteMetric

def WTSNE(dataset, datasetName: str, model=None):

    X = dataset.np_features.astype(np.float32)
    y = dataset.np_labels

    title = f"t-SNE visualization of {datasetName} without weighing"
    modelName = ''

    if(model):
        minMaxNormalized = min_max_normalized_weights(model)
        X *= minMaxNormalized.detach().cpu().numpy()
        modelName = model.name
        title = f"Weighted {datasetName} feature vectors using the {modelName}"
    
    two_color_cmap = ListedColormap(['#1f77b4', '#ff0000'])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)

    print(f"plotting {modelName}...")
    X_tsne = tsne.fit_transform(X)

    datapath = f'results/{datasetName}{modelName}'
    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=two_color_cmap, alpha=0.5)
    plt.colorbar(scatter, label="Label")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.savefig(datapath)
    #plt.show()

    silhouetteMetric(X_tsne, y)

    return X_tsne, y