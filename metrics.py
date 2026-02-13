import numpy as np

def confusion_matrix(actual, predicted, labels):
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for a, p in zip(actual, predicted):
        i = labels.index(a)
        j = labels.index(p)
        matrix[i][j] += 1

    return matrix
def precision_recall_f1(cm):
    precision = []
    recall = []
    f1 = []

    for i in range(len(cm)):
        tp = cm[i][i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp

        p = tp / (tp + fp) if (tp + fp) != 0 else 0
        r = tp / (tp + fn) if (tp + fn) != 0 else 0
        f = 2*p*r/(p+r) if (p+r) != 0 else 0

        precision.append(p)
        recall.append(r)
        f1.append(f)

    return precision, recall, f1

