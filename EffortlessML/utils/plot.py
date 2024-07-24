import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_cv_roc(fpr_list, tpr_list, roc_auc_list):
    # Calculate mean FPR and TPR across all folds
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for i in range(len(fpr_list)):
        tprs.append(np.interp(mean_fpr, fpr_list[i], tpr_list[i]))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0


    # Plot ROC curve
    plt.figure(figsize=(8, 6))

    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

    # Plot mean ROC curve
    sns.lineplot(x=mean_fpr, y=mean_tpr, color='black', label=f'Mean ROC (AUC = {np.mean(roc_auc_list):.2f})')

    # Plot individual ROC curves
    for i in range(len(fpr_list)):
        sns.lineplot(x=fpr_list[i], y=tpr_list[i], alpha=0.1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_cv_cm(confusion_matrix_list):
    plt.figure(figsize=(5, 4))
    sns.heatmap(np.sum(confusion_matrix_list, axis=0), annot=True, fmt='d', cmap='Blues')
    plt.title('Average Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
