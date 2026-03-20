from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_probs)
    return acc, roc