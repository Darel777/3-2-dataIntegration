import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from matplotlib import MatplotlibDeprecationWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def accuracy_custom(y_true, y_pred, threshold=1):
    """
    自定义准确率函数，当预测值和真实值之间的差小于等于threshold时，认为预测正确
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param threshold: 误差阈值，默认为1
    :return: 准确率
    """
    n_samples = len(y_true)
    n_correct = sum(abs(y_true - y_pred) <= threshold)
    accuracy = n_correct / n_samples
    return accuracy


def plot_confusion_matrix(y_true, y_pred, classes, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    title = title
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_norm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, format(cm_norm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

    print('------------------------------------------------------------------------------------------------star_level------------------------------------------------------------------------------------------------')
    df1 = pd.read_csv('star_after_pca.csv')
    co_list1 = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'sa_bal', 'fin_bal',
                'sa_td_bal', 'ntc_bal', 'td_3m_bal', 'td_6m_bal', 'td_1y_bal', 'td_2y_bal', 'pca_1', 'pca_2', 'pca_3']
    x_train, x_test, y_train, y_test = train_test_split(df1[co_list1], df1['star_level'], test_size=0.1, random_state=0)

    lr = LogisticRegression(max_iter=500)  # 迭代到10000也只有0.02%的概率差
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_lr)
    accuracy2 = accuracy_custom(y_test, y_pred_lr)
    precision = precision_score(y_test, y_pred_lr, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_lr, average='macro')
    f1 = f1_score(y_test, y_pred_lr, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_lr)
    print('线性回归模型——误差为0准确率为：', accuracy, '，误差为1准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_dt)
    accuracy2 = accuracy_custom(y_test, y_pred_dt)
    precision = precision_score(y_test, y_pred_dt, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_dt, average='macro')
    f1 = f1_score(y_test, y_pred_dt, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_dt)
    print('决策树模型——误差为0准确率为：', accuracy, '，误差为1准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    accuracy2 = accuracy_custom(y_test, y_pred_rf)
    precision = precision_score(y_test, y_pred_rf, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_rf, average='macro')
    f1 = f1_score(y_test, y_pred_rf, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_rf)
    print('随机森林模型——误差为0准确率为：', accuracy, '，误差为1准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    xgb = XGBClassifier()
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    xgb.fit(x_train, y_train)
    y_pred_xgb = xgb.predict(x_test)
    y_pred_xgb += 1
    accuracy = accuracy_score(y_test, y_pred_xgb)
    accuracy2 = accuracy_custom(y_test, y_pred_xgb)
    precision = precision_score(y_test, y_pred_xgb, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_xgb, average='macro')
    f1 = f1_score(y_test, y_pred_xgb, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_xgb)
    print('XGBoost模型——误差为0准确率为：', accuracy, '，误差为1准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    w1 = 0.8278268360973509 / (0.8278268360973509 + 0.8828774499246177 + 0.9021753176825329 + 0.9066982554382942)
    w2 = 0.8828774499246177 / (0.8278268360973509 + 0.8828774499246177 + 0.9021753176825329 + 0.9066982554382942)
    w3 = 0.9021753176825329 / (0.8278268360973509 + 0.8828774499246177 + 0.9021753176825329 + 0.9066982554382942)
    w4 = 0.9066982554382942 / (0.8278268360973509 + 0.8828774499246177 + 0.9021753176825329 + 0.9066982554382942)
    y_pred_weighted = (w1 * y_pred_lr + w2 * y_pred_dt + w3 * y_pred_rf + w4 * y_pred_xgb)
    nearest_idx = np.abs(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) - y_pred_weighted.reshape(-1, 1)).argmin(axis=1)
    y_pred_weighted = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])[nearest_idx]
    accuracy = accuracy_score(y_test, y_pred_weighted)
    accuracy2 = accuracy_custom(y_test, y_pred_weighted)
    precision = precision_score(y_test, y_pred_weighted, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_weighted, average='macro')
    f1 = f1_score(y_test, y_pred_weighted, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_weighted)
    print('加权模型——误差为0准确率为：', accuracy, '，误差为1准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    save_path = 'confusion_matrix_star.png'
    title = 'confusion_matrix_star'
    class_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    plot_confusion_matrix(y_test, y_pred_weighted, class_names, title, save_path)

    print('------------------------------------------------------------------------------------------------credit_level------------------------------------------------------------------------------------------------')
    df2 = pd.read_csv('credit_after_pca.csv')
    co_list2 = ['sex', 'marriage', 'education', 'is_black', 'is_contact', 'all_bal', 'due_intr', 'bad&delay']
    x_train, x_test, y_train, y_test = train_test_split(df2[co_list2], df2['credit_level'], test_size=0.1, random_state=0)

    lr = LogisticRegression(max_iter=500)  # 迭代到10000也只有0.02%的概率差
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_lr)
    accuracy2 = accuracy_custom(y_test, y_pred_lr, 15)
    precision = precision_score(y_test, y_pred_lr, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_lr, average='macro')
    f1 = f1_score(y_test, y_pred_lr, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_lr)
    print('线性回归模型——误差为0准确率为：', accuracy, '，误差为15准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred_dt = dt.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_dt)
    accuracy2 = accuracy_custom(y_test, y_pred_dt, 15)
    precision = precision_score(y_test, y_pred_dt, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_dt, average='macro')
    f1 = f1_score(y_test, y_pred_dt, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_dt)
    print('决策树模型——误差为0准确率为：', accuracy, '，误差为15准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    accuracy2 = accuracy_custom(y_test, y_pred_rf, 15)
    precision = precision_score(y_test, y_pred_rf, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_rf, average='macro')
    f1 = f1_score(y_test, y_pred_rf, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_rf)
    print('随机森林模型——误差为0准确率为：', accuracy, '，误差为15准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    xgb = XGBClassifier()
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    xgb.fit(x_train, y_train)
    y_pred_xgb = xgb.predict(x_test)
    y_pred_xgb += 1
    y_pred_xgb = np.where(y_pred_xgb == 1, 35, y_pred_xgb)
    y_pred_xgb = np.where(y_pred_xgb == 2, 50, y_pred_xgb)
    y_pred_xgb = np.where(y_pred_xgb == 3, 60, y_pred_xgb)
    y_pred_xgb = np.where(y_pred_xgb == 4, 70, y_pred_xgb)
    y_pred_xgb = np.where(y_pred_xgb == 5, 85, y_pred_xgb)
    accuracy = accuracy_score(y_test, y_pred_xgb)
    accuracy2 = accuracy_custom(y_test, y_pred_xgb, 15)
    precision = precision_score(y_test, y_pred_xgb, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_xgb, average='macro')
    f1 = f1_score(y_test, y_pred_xgb, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_xgb)
    print('XGBoost模型——误差为0准确率为：', accuracy, '，误差为15准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    w1 = 0.7286135693215339 / (0.7286135693215339 + 0.8964274008521796 + 0.8970829236315963 + 0.9062602425434284)
    w2 = 0.8964274008521796 / (0.7286135693215339 + 0.8964274008521796 + 0.8970829236315963 + 0.9062602425434284)
    w3 = 0.8970829236315963 / (0.7286135693215339 + 0.8964274008521796 + 0.8970829236315963 + 0.9062602425434284)
    w4 = 0.9062602425434284 / (0.7286135693215339 + 0.8964274008521796 + 0.8970829236315963 + 0.9062602425434284)
    y_pred_weighted = (w1 * y_pred_lr + w2 * y_pred_dt + w3 * y_pred_rf + w4 * y_pred_xgb)
    nearest_idx = np.abs(np.array([35, 50, 60, 70, 85]) - y_pred_weighted.reshape(-1, 1)).argmin(axis=1)
    y_pred_weighted = np.array([35, 50, 60, 70, 85])[nearest_idx]
    accuracy = accuracy_score(y_test, y_pred_weighted)
    accuracy2 = accuracy_custom(y_test, y_pred_weighted, 15)
    precision = precision_score(y_test, y_pred_weighted, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred_weighted, average='macro')
    f1 = f1_score(y_test, y_pred_weighted, average='macro')
    kappa = cohen_kappa_score(y_test, y_pred_weighted)
    print('加权模型——误差为0准确率为：', accuracy, '，误差为15准确率为：', accuracy2, '，精确率为：', precision, '，召回率为：', recall, '，F1分数为：', f1, '，Kappa系数为：', kappa, "。")

    save_path = 'confusion_matrix_credit.png'
    title = 'confusion_matrix_credit'
    class_names = ['35', '50', '60', '70', '85']
    plot_confusion_matrix(y_test, y_pred_weighted, class_names, title, save_path)


if __name__ == '__main__':
    main()
