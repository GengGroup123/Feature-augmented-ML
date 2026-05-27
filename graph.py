import pandas as pd
import seaborn as sns
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
import numpy as np

# 绘制特征相关性热图
def pearson(x,color=True):
    # Covariance of every feature
    plt.figure(figsize=(10, 10))
    sns.heatmap(x.corr(), annot=color)
    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix (without normalization)'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes)-0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax



def ROC_curve(y_true, y_pred,sample_weight0=1,save = False,name = '\\roc_descriptor.xlsx'):

    # sample_weight0 is the num of true class weight for every data

    y_label = y_true
    y_pre = y_pred

    sample_weights = [sample_weight0 if label == 1 else 1 for label in y_label]

    fpr, tpr, thresholds = roc_curve(y_label, y_pre, sample_weight=sample_weights)

    cha = max(tpr-fpr)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds[maxindex]
    print('when tpr-fpr is max, the threshold is ' + str(threshold))



    roc_auc = auc(fpr, tpr)

    plt.scatter(fpr[maxindex], tpr[maxindex], color='red', marker='x',
                label=f'Best Threshold (tpr-fpr = {cha:.2f})')

    plt.plot(fpr, tpr, 'm--', label='(AUC = {0:.2f})'.format(roc_auc), lw=2)
    plt.plot([-0.05, 1.05], [-0.05, 1.05], 'k-', label='Random Guess (y=x)', lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    if save == True:
        print(fpr)
        print(tpr)
        table1 = pd.DataFrame(fpr)
        table2 = pd.DataFrame(tpr)
        table = pd.concat([table1, table2], axis=1)
        table.columns = ['fpr', 'tpr']
        print(table)
        table.to_excel(r"C:\Users\hp\Desktop" + name)



def PR_curve(y_true, y_pred, sample_weight0=1,save = False,name = '\\pr_descriptor.xlsx'):


    y_label = y_true  # 真实标签
    y_pre = y_pred  # 模型预测的概率值（通常是正类的概率）


    # 计算精确率（Precision）和召回率（Recall），以及决策阈值
    sample_weights = [sample_weight0 if label == 1 else 1 for label in y_label]
    precision, recall, thresholds = precision_recall_curve(y_label, y_pre,sample_weight=sample_weights)

    # 计算每个阈值对应的 F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # 找到 F1-score 最大的阈值
    best_threshold_index = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_index]
    best_f1_score = f1_scores[best_threshold_index]

    # 输出最佳阈值和对应的 F1-score
    print("Best Threshold: ", best_threshold)
    print("Best F1-score: ", best_f1_score)

    # 绘制 PR 曲线
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, 'b-', label='(AUC = {0:.2f})'.format(pr_auc), lw=2)

    # 标记最佳阈值
    plt.scatter(recall[best_threshold_index], precision[best_threshold_index], color='red', marker='x',
                label=f'Best Threshold (F1 = {best_f1_score:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])

    if save == True:
        print(recall)
        print(precision)
        table1 = pd.DataFrame(recall)
        table2 = pd.DataFrame(precision)
        table = pd.concat([table1, table2], axis=1)
        table.columns = ['recall', 'precision']
        print(table)
        table.to_excel(r"C:\Users\hp\Desktop" + name)


