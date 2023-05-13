#!/usr/bin/env python
# coding: utf-8

#  Copyright (c) 2023 DZX.
#
#  All rights reserved.
#
# This software is protected by copyright law and international treaties. No part of this software may be reproduced,
# distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or
# mechanical methods, without the prior written permission of the copyright owner.
#
#  For permission requests, please contact the copyright owner at the address below.
#
#  DZX
#
#  xindemicro@outlook.com
#

import pickle
import warnings

# 导入相关库
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

warnings.filterwarnings("ignore")


def download_nltk_corpus(corpus_name):
    """
    下载 NLTK 语料库，如果语料库尚未下载
    :param corpus_name: 语料库名字
    :return:
    """
    try:
        # 检查语料库是否已下载
        nltk.data.find(corpus_name)
    except LookupError:
        # 如果语料库尚未下载，则下载语料库
        nltk.download(corpus_name)


# download_nltk_corpus('stopwords')
# download_nltk_corpus('punkt')

stop_words = list(stopwords.words("english"))


# 加载特征数据的函数，输入为数据目录、特征名和样本数，返回特征数据
def load_features(data_dir, feature, n=-1):
    """
    加载特征文件
    :param data_dir:str，特征数据存储目录
    :param feature:str，特征类型，可选的值为 "count" 和 "tfidf"
    :param n:int，要加载的特征数量，默认为全部加载
    :return: data: numpy.ndarray，特征数据
    """
    # 获取数据存储目录的信息，例如从"from_feature"提取"from"
    # print(data_dir)
    info = data_dir[12:-8]
    # print(info)
    # 如果数据存储目录为"from_feature"或"to_feature"，将其改为"fromto_feature"
    if data_dir == "../features/from_feature" or data_dir == "../features/to_feature":
        data_dir = "../features/fromto_feature"
    # 构造特征数据的文件名
    features_filename = f"{data_dir}/{info}_{feature}_features.p"
    # 从文件中加载特征数据
    data = pickle.load(open(features_filename, "rb"))
    # 如果指定了要加载的特征数量，则只加载前n个特征；否则加载全部特征
    return data[:n] if n != -1 else data


# 定义加载数据的方法
def load_labels(data_dir, n=-1):
    """
    读取二进制文件中的数据，并返回数据前n个或全部数据。
    :param data_dir:str，要读取的文件路径。
    :param n:int，要读取的文件路径。
    :return:data (obj): 读取的数据。
    """
    data = pickle.load(open(f"../features/{data_dir}/labels.p", "rb"))
    return data[:n] if n != -1 else data


def structural_model():
    """
    实现基于邮件结构数据的结构模型
    :return:无，直接输出最佳的超参数 alpha、测试集上的 f1 值、测试集上的准确率
    """
    print("structural_model:")
    # 读取邮件结构数据
    structural_info = pd.read_csv("../data/email_structure_full.csv")
    # 将数据集划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(structural_info[[
        "has_multipart", "has_html", "has_links", "has_attachments"
    ]],
                                                        structural_info["spam"],
                                                        shuffle=True,
                                                        random_state=42,
                                                        test_size=0.15)
    # 交叉验证
    alpha_list = [0.1, 1, 10, 100, 1000, 2000, 5000, 10000]
    # 找到f1最大的alpha值
    best_alpha = alpha_list[np.argmax([
        f1_score(y_train, np.argmax(MultinomialNB(alpha=a).fit(x_train, y_train).predict_proba(x_train), axis=1),
                 average='weighted')
        for a in alpha_list
    ])]
    # 拟合模型，使用训练数据训练模型
    model = MultinomialNB(alpha=best_alpha).fit(x_train, y_train)
    # 对测试集进行预测并计算性能指标
    y_pred = np.argmax(model.predict_proba(x_test), axis=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = np.mean(model.predict(x_test) == y_test)
    print(f"最佳的超参数alpha：{best_alpha}\n"
          f"测试集上的f1值：{f1}\n"
          f"测试集上的准确率：{accuracy}")


def subject_model(feature="count", n=-1, model_type="MB"):
    """
    实现基于主题数据的主题模型
    :param feature:str，特征类型，可选的值为 "count" 和 "tfidf"
    :param n:int，特征数量，默认为-1，表示使用全部特征
    :param model_type: str，分类器类型，可选的值为 "MB" 和 "CB"，分别代表MultinomialNB和ComplementNB
    :return:tuple，元素分别为训练集的特征、测试集的特征、训练集的标签、测试集的标签
    """
    print("subject_model:")
    # 设置数据存储目录
    data_dir = "../features/subject_feature"
    # 加载特征文件和标签文件
    X_set = load_features(data_dir, feature)
    labels = load_labels(data_dir, n)
    # 将数据集划分为训练集和测试集，其中测试集占比为0.15
    x_train, x_test, y_train, y_test = train_test_split(X_set, labels, shuffle=True, random_state=42, test_size=0.15)
    # 设置待优化的超参数alpha的候选值
    # alpha_list = [0.1, 1, 10, 100, 1000]
    # 设置网格搜索的参数
    parameters = {
        'alpha': [0.1, 1, 10, 100, 1000],
        'fit_prior': [True, False]
    }
    classifier = MultinomialNB if model_type == "MB" else ComplementNB
    # 完成模型的交叉验证，得到f1和准确率结果
    grid_search = GridSearchCV(classifier(), parameters, cv=5,
                               scoring={'f1_weighted', 'accuracy'}, refit='f1_weighted')
    grid_search.fit(x_train, y_train)
    # 获取最佳的alpha值
    best_alpha = grid_search.best_params_['alpha']
    best_fit_prior = grid_search.best_params_['fit_prior']
    print(grid_search.best_params_)
    # 输出相关信息
    print(f"{feature} 特征： ")
    print("最佳alpha值：", best_alpha)
    print("NB的5折交叉验证的最大f1值：", grid_search.best_score_)
    # 使用最佳的alpha值训练模型
    model = classifier(alpha=best_alpha, fit_prior=best_fit_prior)
    model.fit(x_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(x_test)
    # 输出测试集上的f1值和准确率
    print("测试集上的f1值：", f1_score(y_test, y_pred, average='weighted'))
    print("测试集上的accuracy值：", np.mean(model.predict(x_test) == y_test))
    return x_train, x_test, y_train, y_test


def message_model(feature="count", n=-1, model_type="MB"):
    """
    实现基于邮件正文数据的邮件正文模型
    :param feature:str，特征类型，可选的值为 "count" 和 "tfidf"
    :param n:int，特征数量，默认为-1，表示使用全部特征
    :param model_type: str，分类器类型，可选的值为 "MB" 和 "CB"，分别代表MultinomialNB和ComplementNB
    :return:tuple，元素分别为训练集的特征、测试集的特征、训练集的标签、测试集的标签
    """
    print("message_model:")
    # 设置数据存储目录
    data_dir = "../features/message_feature"
    # 加载特征文件和标签文件
    X_set = load_features(data_dir, feature)
    labels = load_labels(data_dir, n)
    # 将数据集划分为训练集和测试集，其中测试集占比为0.15
    x_train, x_test, y_train, y_test = train_test_split(X_set, labels, shuffle=True, random_state=42, test_size=0.15)
    # 设置待优化的超参数alpha的候选值
    alpha_list = [0.1, 1, 10, 100, 1000]
    # 根据model_type选择分类器类型，MultinomialNB或ComplementNB
    classifier = MultinomialNB if model_type == "MB" else ComplementNB
    model = GridSearchCV(classifier(), param_grid={'alpha': alpha_list}, cv=5,
                         scoring=['f1_weighted', 'accuracy'], refit='f1_weighted')
    model.fit(x_train, y_train)
    best_alpha = model.best_params_['alpha']
    # 输出相关信息
    print(f"{feature} 特征：")
    print("最佳alpha值：", best_alpha)
    print("NB的5折交叉验证的最大f1值：", model.best_score_)
    # 使用最佳的alpha值训练模型
    model = classifier(alpha=best_alpha)
    model.fit(x_train, y_train)
    # 对测试集进行预测
    y_pred = model.predict(x_test)
    # 输出测试集上的f1值和准确率
    print("测试集上的f1值：", f1_score(y_test, y_pred, average='weighted'))
    print("测试集上的accuracy值：", np.mean(model.predict(x_test) == y_test))
    return x_train, x_test, y_train, y_test


# 定义函数combined_model，参数包括特征(feature)类型，样本数(n)，模型类型(model_type)
def comprehensive_model(feature="count", n=-1, model_type="MB"):
    """
    实现综合特征模型的训练与评估。
    通过使用多种特征（包括邮件内容、邮件主题、邮件发件人、邮件收件人、邮件结构等），并使用GridSearchCV实现交叉验证，确定最佳的超参数alpha。
    最后根据最佳的alpha值训练模型，并对测试集进行预测评估。
    :param feature:特征名称，可选'count'或'tfidf'
    :param n:训练数据量，默认-1表示全部数据，否则选取前n条数据
    :param model_type:分类器类型，可选"MB"（MultinomialNB）或"CB"（ComplementNB）
    :return:训练集和测试集数据
    """
    print("comprehensive_model:")
    # 定义数据目录和特征名列表
    data_dirs = [
        "../features/message_feature", "../features/subject_feature", "../features/from_feature", "../features/to_feature"
    ]

    # data_dirs = [
    #     "message_feature"
    # ]

    features = [feature, feature, f"{feature}", f"{feature}"]

    # features = [feature]

    # 使用列表推导式和zip函数加载特征数据
    X_sets = [
        load_features(data_dir, feat, n)
        for data_dir, feat in zip(data_dirs, features)
    ]
    # 加载标签数据
    # labels_filename = "message_feature/labels.p"
    labels = load_labels("message_feature", n)
    # 读取邮件结构信息，将其转化成稀疏矩阵
    structural_info = pd.read_csv("../data/email_structure_full.csv").iloc[:, :-1]
    structural_info = structural_info.values[:n] if n != -1 else structural_info.values
    structural_info = csr_matrix(structural_info)
    print(len(X_sets), structural_info.shape)
    # 水平拼接特征数据，构建输入数据矩阵
    X_set = hstack(X_sets + [structural_info])
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X_set,
                                                        labels,
                                                        shuffle=True,
                                                        random_state=42,
                                                        test_size=0.15)
    # 定义alpha值列表，用于尝试不同的alpha值进行训练
    alpha_list = [0.1, 1, 10, 100, 1000]
    # 根据model_type选择分类器类型，MultinomialNB或ComplementNB
    classifier = MultinomialNB if model_type == "MB" else ComplementNB
    # 使用列表推导式进行交叉验证
    results = [
        cross_validate(classifier(alpha=a),
                       x_train,
                       y_train,
                       scoring=["f1_weighted", "accuracy"],
                       cv=5,
                       return_train_score=True) for a in alpha_list
    ]
    # 提取交叉验证的f1值和准确率
    f1_list = [result['test_f1_weighted'].mean() for result in results]
    accuracy_list = [result['test_accuracy'].mean() for result in results]
    # 获取最佳alpha值
    best_alpha = alpha_list[np.argmax(f1_list)]
    # 输出相关信息
    print(f"{feature} 特征：")
    print(f1_list)
    print("最佳alpha值：", best_alpha)
    print("NB的5折交叉验证的最大f1值：", np.max(f1_list))
    print("NB的5折交叉验证的最大accuracy值：", accuracy_list[np.argmax(f1_list)])
    # 选择5折交叉验证 f1最大对应的alpha值重新训练模型，并在测试集上评估
    model = classifier(alpha=1)
    model.fit(x_train, y_train)
    # 保存训练好的模型
    with open(f"../models/CombinedFeatNB_{feature}.p", "wb") as f:
        pickle.dump(model, f)
    # 在测试集上进行预测并输出结果
    Y_pred = model.predict_proba(x_test)
    y_pred = np.argmax(Y_pred, axis=1)
    # y_pred = (Y_pred[:, 1] > 0.5).astype(int)
    # 打印测试集上的f1值和准确率
    print("测试集上的f1值：", f1_score(y_test, y_pred, average='weighted'))
    print("测试集上的accuracy值：", np.mean(model.predict(x_test) == y_test))
    # 返回训练集和测试集数据
    return x_train, x_test, y_train, y_test


# 定义 CombinedNB 分类器
class CombinedNB(BaseEstimator):
    """
    组合多项式朴素贝叶斯分类器

    :param
    ----------
    n1 : int，第一个MultinomialNB分类器使用的特征数量
    n2 : int，第二个MultinomialNB分类器使用的特征数量
    n3 : int，第三个MultinomialNB分类器使用的特征数量
    a : float，平滑参数，用于处理零频率

    :member
    ----------
    model1 : MultinomialNB，第一个MultinomialNB分类器
    model2 : MultinomialNB，第二个MultinomialNB分类器
    model3 : MultinomialNB，第三个MultinomialNB分类器
    class_prior_ : ndarray, shape (n_classes,)，每个类别的先验概率

    :method
    -------
    fit(X, y)，用于训练模型，X是训练数据，y是训练标签
    predict_proba(X)，用于预测样本属于不同类别的概率，X是测试数据
    predict(X)，用于预测样本所属类别，X是测试数据
    """

    # 初始化方法，n1、n2、n3、a是类的成员变量
    def __init__(self, n1, n2, n3, a):
        """
        初始化方法，创建类的成员变量
        """
        self.n1, self.n2, self.n3, self.a = n1, n2, n3, a

    # 拟合方法，对每个子模型进行训练
    # fit方法，用于训练模型，X是训练数据，y是训练标签
    def fit(self, X, y):
        # 创建一个MultinomialNB分类器，指定平滑参数alpha，使用前n1个特征训练MultinomialNB分类器
        self.model1 = MultinomialNB(alpha=self.a).fit(X[:, :self.n1], y)
        # 计算每个类别的先验概率
        self.class_prior_ = self.model1.class_count_ / self.model1.class_count_.sum(
        )
        # 创建另一个MultinomialNB分类器，指定平滑参数alpha，使用第n1个特征到第(n1+n2)个特征训练MultinomialNB分类器
        self.model2 = MultinomialNB(alpha=self.a).fit(
            X[:, self.n1:(self.n1 + self.n2)], y)
        # 创建另一个MultinomialNB分类器，指定平滑参数alpha，使用第(n1+n2)个特征到第(n1+n2+n3)个特征训练MultinomialNB分类器
        self.model3 = MultinomialNB(alpha=self.a).fit(
            X[:, (self.n1 + self.n2):(self.n1 + self.n2 + self.n3)], y)

    # 预测概率方法，结合子模型的概率
    # predict_proba方法，用于预测样本属于不同类别的概率，X是测试数据
    def predict_proba(self, X):
        # 分别对前n1个特征、第n1个特征到第(n1+n2)个特征、第(n1+n2)个特征到第(n1+n2+n3)个特征计算各个类别的概率
        p1, p2, p3 = self.model1.predict_proba(
            X[:, :self.n1]), self.model2.predict_proba(
            X[:, self.n1:(self.n1 + self.n2)]), self.model3.predict_proba(
            X[:, (self.n1 + self.n2):(self.n1 + self.n2 + self.n3)])
        # 计算最终概率，使用了类先验概率
        p = p1 * p2 * p3 / self.class_prior_
        # 对概率进行归一化
        return p / p.sum(axis=1)[:, np.newaxis]

    # 预测方法
    # predict方法，用于预测样本所属类别，X是测试数据
    def predict(self, X):
        # 预测分类结果
        return np.argmax(self.predict_proba(X), axis=1)


# 定义 combined_model 方法，用于训练并评估模型
def combined_model(feature="count", n=-1):
    """
    训练朴素贝叶斯模型，并评估其准确率
    :param feature:str, default="count"，特征类型，可选择 "count" 或 "tfidf"
    :param n:int, default=-1，训练数据的数量，-1 表示使用全部数据，默认值为-1
    :return:
    -------
    x_train: csr_matrix，训练数据的特征
    x_test: csr_matrix，测试数据的特征
    y_train: array，训练数据的标签
    y_test: array，测试数据的标签
    """
    # 加载数据
    print("combined_model:")
    X_body, X_sub = load_features("../features/message_feature", feature), load_features("../features/subject_feature", feature)
    # X_body, X_sub = load_data(
    #     "message_feature/message_{}_features.p".format(feature)), load_data(
    #     "subject_feature/subject_{}_features.p".format(feature))
    n_body, n_sub = X_body.shape[1], X_sub.shape[1]
    # 加载结构信息数据
    structural_info = pd.read_csv("../data/email_structure_full.csv").iloc[:, :-1].values
    structural_info = csr_matrix(
        structural_info[:n]) if n != -1 else csr_matrix(structural_info)
    n_structural = structural_info.shape[1]
    print(X_body.shape, X_sub.shape, structural_info.shape)
    # 合并特征和标签
    X_set, labels = hstack([X_body, X_sub, structural_info
                            ]), load_labels("message_feature", n)
    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X_set,
                                                        labels,
                                                        shuffle=True,
                                                        random_state=42,
                                                        test_size=0.15)
    # 调优alpha参数
    alpha_list = [0.1, 1, 10, 100, 1000]
    results = [
        cross_validate(CombinedNB(n_body, n_sub, n_structural, a),
                       x_train,
                       y_train,
                       scoring=["f1_weighted", "accuracy"],
                       cv=5,
                       return_train_score=True) for a in alpha_list
    ]
    # 计算 F1 和准确率
    f1_list = [result['test_f1_weighted'].mean() for result in results]
    accuracy_list = [result['test_accuracy'].mean() for result in results]
    # 找到最佳 alpha
    best_alpha = alpha_list[np.argmax(f1_list)]
    # 输出结果
    print("{} 特征：".format(feature))
    print(f1_list)
    print("最佳alpha值：", best_alpha)
    print("NB的5折交叉验证的最大f1值：", np.max(f1_list))
    print("NB的5折交叉验证的最大accuracy值：", accuracy_list[np.argmax(f1_list)])
    # 使用最佳 alpha 训练模型
    model = CombinedNB(n_body, n_sub, n_structural, a=best_alpha)
    model.fit(x_train, y_train)
    # 保存模型
    with open("../models/CombinedProbNB_{}.p".format(feature), "wb") as f:
        pickle.dump(model, f)
    # 在测试集上评估模型
    test_accuracy = np.mean(model.predict(x_test) == y_test)
    print("测试集上的accuracy值：", test_accuracy)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    print("----------------------------------------------------------------")
    structural_model()

    # 依次对三种不同特征进行模型评估
    print("----------------------------------------------------------------")
    subject_model(feature="count")
    print("----------------------------------------------------------------")
    subject_model(feature="tfidf")
    # print("----------------------------------------------------------------")

    # 对于不同的 feature 和 model_type，分别调用 message_model 函数，并输出结果
    print("----------------------------------------------------------------")
    message_model(feature="count", n=-1)
    print("----------------------------------------------------------------")
    message_model(feature="tfidf", n=-1)
    # print("----------------------------------------------------------------")

    # 分别对三种特征进行模型训练和测试，并输出结果
    print("----------------------------------------------------------------")
    comprehensive_model(feature="count", n=-1)
    print("----------------------------------------------------------------")
    comprehensive_model(feature="tfidf", n=-1)
    # print("----------------------------------------------------------------")

    # 使用三种特征分别调用combined_model2函数进行训练和测试
    print("----------------------------------------------------------------")
    combined_model(feature="count", n=-1)
    print("----------------------------------------------------------------")
    combined_model(feature="tfidf", n=-1)
    # print("----------------------------------------------------------------")
