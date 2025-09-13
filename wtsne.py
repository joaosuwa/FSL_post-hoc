import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from utils import get_feature_weights_as_numpy, get_feature_weights_as_tensor, min_max_normalized_weights
import numpy as np
import matplotlib.pyplot as plt
from openTSNE import TSNE as open_TSNE
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

def WTSNEv2(logger, X, y, model=None, weights=None, name=None, f_scaler=StandardScaler, n_components=2, perplexity=30):
    title = f"t-SNE visualization without weighing"
    X = X.copy()
    X = f_scaler().fit_transform(X)

    if model is not None:
        weights = get_feature_weights_as_tensor(model)

    if weights is not None:
        weights = weights.detach().cpu().numpy().reshape(-1, 1)

    if weights is not None:
        title = f"Weighted feature vectors"
        W = f_scaler().fit_transform(weights)
        W = np.tile(W,(1, X.shape[0])).transpose()
        X = np.multiply(W,X)

    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=42,
        n_iter=500,
        verbose=1
    )

    embedding = tsne.fit_transform(X)

    embedding_silhouette = metrics.silhouette_score(embedding, y, metric='euclidean')

    datapath = f'{logger.dir_path}/tsne_plot{f"_{name}" if name else ""}.pdf'

    colors = [
    '#FF0000',  # Pure Red
    '#0000FF',  # Pure Blue
    '#008000',  # Green
    '#FFA500',  # Orange
    '#800080',  # Purple
    '#00FFFF',  # Cyan
    '#FFC0CB',  # Pink
    '#A52A2A',  # Brown
    '#808000',  # Olive
    '#000000'   # Black
    ]

    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'X']

    # Plot
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(y)):
        idx = y == label
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            alpha=0.6,
            label=None  # No legend
        )

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.savefig(datapath)
    plt.close()
    #plt.show()

    logger.log_text(f"Embedding silhouette {name}: {embedding_silhouette}.")

    return embedding_silhouette
