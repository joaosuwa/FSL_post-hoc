from metrics import informativeFeaturesMetric
import matplotlib.pyplot as plt

def PIFS_PSFI(executionId, models, feature_cols, datasetName: str):
    results = []

    for model in models:
        scores = informativeFeaturesMetric(model, feature_cols)
        results.append((model.name, scores))

    plt.figure(figsize=(8, 6))
    for model_name, scores in results:
        x = list(range(1, len(scores) + 1))
        y_PIFS = [score['PIFS'] for score in scores]
        y_PSFI = [score['PSFI'] for score in scores]

        plt.plot(x, y_PIFS, ':',label=f"{model_name} - PIFS")
        plt.plot(x, y_PSFI, '--',label=f"{model_name} - PSFI")
 
    datapath = f'results/{executionId}/{datasetName} Informative Features Metric'

    plt.xlabel('Number of features selected')
    plt.ylabel('Percentage')
    plt.title(f'Percentage of informative features in {datasetName} dataset')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(datapath)
    #plt.show()