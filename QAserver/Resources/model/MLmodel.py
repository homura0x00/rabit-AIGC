import os
import pickle
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 读取数据
def read_data(data_path):

    X, y = [], []
    with open(data_path, 'r', encoding='utf8') as f:
        for line in f.readlines():
            # 划分样本和标签
            text, label = line.strip().split(',')
            text = ' '.join(list(text.lower()))
            # 存储
            X.append(text)
            y.append(label)

    index = np.arange(len(X))
    np.random.shuffle(index)
    X = [X[i] for i in index]
    # print(*X, sep='\n')
    y = [y[i] for i in index]
    set_y = set(y)
    # print("样本个数：", len(X))
    # print("标签有：", set_y)
    # print("标签个数为：", len(set_y))
    return X, y

# , def run(data_path,model_save_path):


def run(data_path, model_save_path):
    X, y = read_data(data_path)
    # 计算数据集中共有多少标签
    label_set = sorted(list(set(y)))
    # 使用阿拉伯数字0-5对label进行再标签
    label_id = {label: idx for idx, label in enumerate(label_set)}
    id2label = {idx: label for label, idx in label_id.items()}

    # 以阿拉伯数字形式来代替字符串标签
    y = [label_id[i] for i in y]

    label_names = sorted(label_id.items(), key=lambda kv: kv[1], reverse=False)
    # 预测模型的参数
    target_names = [i[0] for i in label_names]
    # 混淆矩阵的参数
    labels = [i[1] for i in label_names]

    # 划分训练集和测试集
    train_X, text_X, train_y, text_y = train_test_split(
        X, y, test_size=0.15, random_state=42)

    # 机器学习的文本特征提取，文本类数据向量化，
    txtvector = TfidfVectorizer(ngram_range=(1, 3), min_df=0, max_df=0.9,
                                analyzer='char', use_idf=1, smooth_idf=1, sublinear_tf=1)
    train_X = txtvector.fit_transform(train_X)
    text_X = txtvector.transform(text_X)

    # -------------LR--------------
    print("\n------------------ LR --------------------")
    print("\nLR训练中....")
    LR = LogisticRegression(C=5, dual=False, n_jobs=4, solver='sag',
                            max_iter=200, multi_class='ovr', random_state=124)
    LR.fit(train_X, train_y)
    # 计算预测值
    pred = LR.predict(text_X)
    print("LogisticRegression - Classification report\n")
    print(classification_report(text_y, pred, target_names=target_names))
    # 多分类roc_auc_score()函数的值
    proba = LR.predict_proba(text_X)
    auc = roc_auc_score(text_y, proba, multi_class='ovr')
    print('AUC:\t', auc)

    # 混淆矩阵
    print(confusion_matrix(text_y, pred, labels=labels))
    C = confusion_matrix(text_y, pred, labels=labels)
    sns.heatmap(C, annot=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('逻辑回归-混淆矩阵')
    plt.show()

    # -------------Tree---------------
    print("\n------------------ Tree --------------------")
    print("\nTree训练中...")
    print("Tree:\n")
    tree = DecisionTreeClassifier(splitter='random', random_state=122)
    tree.fit(train_X, train_y)
    pred = tree.predict(text_X)
    print(classification_report(text_y, pred, target_names=target_names))

    proba = tree.predict_proba(text_X)
    auc = roc_auc_score(text_y, proba, multi_class='ovr')
    print('AUC:\t', auc)
    C = confusion_matrix(text_y, pred, labels=labels)
    sns.heatmap(C, annot=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('决策树-混淆矩阵')
    plt.show()

    # # -------------svm--------------
    print("\n------------------ svm --------------------")
    print("\nSVM训练中....")
    svc_clf = svm.LinearSVC(tol=0.00001, C=6.0, multi_class='ovr',
                            class_weight='balanced', random_state=122, max_iter=100)
    svc_clf.fit(train_X, train_y)
    pred = svc_clf.predict(text_X)
    print("svm:\n")
    print(classification_report(text_y, pred, target_names=target_names))
    # 多分类roc_auc_score()函数的值
    proba = svc_clf._predict_proba_lr(text_X)
    auc = roc_auc_score(text_y, proba, multi_class='ovr')
    print('AUC:\t', auc)

    # 混淆矩阵
    print(confusion_matrix(text_y, pred, labels=labels))
    C = confusion_matrix(text_y, pred, labels=labels)
    sns.heatmap(C, annot=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('支持向量机-混淆矩阵')
    plt.show()

    # -------------gbdt--------------
    print("\n------------------ gbdt --------------------")
    print("\ngbdt训练中....")
    gbdt = GradientBoostingClassifier(
        n_estimators=250, learning_rate=0.1, random_state=42)
    gbdt.fit(train_X, train_y)
    pred = gbdt.predict(text_X)
    print("GBDT:\n")
    print(classification_report(text_y, pred, target_names=target_names))
    proba = gbdt.predict_proba(text_X)
    auc = roc_auc_score(text_y, proba, multi_class='ovr')
    print('AUC:\t', auc)
    print(confusion_matrix(text_y, pred, labels=labels))
    C = confusion_matrix(text_y, pred, labels=labels)
    sns.heatmap(C, annot=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('梯度提升树-混淆矩阵')
    plt.show()

    # # rf
    # print("\n------------------ rf --------------------")
    print("\nrf训练中....")
    RF = RandomForestClassifier(random_state=66)
    RF.fit(train_X, train_y)
    pred = RF.predict(text_X)
    print(classification_report(text_y, pred, target_names=target_names))
    print(confusion_matrix(text_y, pred, labels=labels))

    # # -------------融合--------------
    # LR_pred = LR.predict_proba(text_X)
    # Tree_pred = tree.predict_proba(text_X)
    # SVM_pred = svc_clf._predict_proba_lr(text_X)
    # GBDT_pred = gbdt.predict_proba(text_X)

    # LR_Tree = LR_pred + Tree_pred
    # LR_SVM = LR_pred + SVM_pred
    # Tree_SVM = Tree_pred + SVM_pred
    # LR_Tree_SVM = LR_pred + Tree_pred + SVM_pred
    # LR_GBDT = LR_pred + GBDT_pred
    # Tree_GBDT = Tree_pred + GBDT_pred

    # print("\n------------------ LR+TREE --------------------")
    # first_pred = np.argmax(LR_Tree/2, axis=1)
    # print(classification_report(text_y, first_pred, target_names=target_names))
    # print(confusion_matrix(text_y, pred, labels=labels))

    # print("\n------------------ LR+gbdt --------------------")
    # five_pred = np.argmax(LR_GBDT/2, axis=1)
    # print(classification_report(text_y, five_pred, target_names=target_names))
    # print(confusion_matrix(text_y, pred, labels=labels))

    # print("\n------------------ Tree+gbdt --------------------")
    # six_pred = np.argmax((LR_SVM + GBDT_pred)/3, axis=1)
    # print(classification_report(text_y, six_pred, target_names=target_names))
    # print(confusion_matrix(text_y, pred, labels=labels))

    # 保存模型
    pickle.dump(id2label, open(os.path.join(
        model_save_path, 'id2label.pkl'), 'wb'))
    pickle.dump(txtvector, open(os.path.join(
        model_save_path, 'vec.pkl'), 'wb'))
    pickle.dump(LR, open(os.path.join(model_save_path, 'LR.pkl'), 'wb'))
    pickle.dump(gbdt, open(os.path.join(model_save_path, 'gbdt.pkl'), 'wb'))


if __name__ == '__main__':
    # run("./data/ml_data.txt")
    run("./data/ml_data.txt", "./model_file/")
