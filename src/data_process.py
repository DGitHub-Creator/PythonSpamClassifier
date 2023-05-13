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

# 导入相关库
import datetime
import email
import logging
import os
import pickle
import re
import string
import time
import warnings
from email.utils import parseaddr
from multiprocessing import cpu_count

import chardet
import dask.dataframe as dd
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from dask.diagnostics import ProgressBar
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 设置数据目录
DATA_DIR = "D:/py/pythonProjectSpam/trec06p/"
email_id_label = -7
columns = [
    "email_id", "parts", "attachments", "html", "from", "to", "subject",
    "body", "links", "spam"
]
email_data = pd.DataFrame(columns=columns)

# 为 pandas 应用程序添加进度条支持
tqdm.pandas()

print("cpu_count: ", cpu_count())

# 配置日志
logging.basicConfig(filename="../logs/data_preprocessing_log.txt",
                    level=logging.INFO,
                    format="%(asctime)s: %(message)s")


def log_message(label_string):
    """
    记录日志消息，同时在控制台输出该消息。
    :param label_string: 输出提示信息
    :return:
    """
    # 获取当前的时间戳
    ts = time.time()
    # 将时间戳转换为日期时间字符串
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S:%f')
    # 打印当前时间戳和标签字符串
    print("{}: {}".format(st, label_string))
    # 使用 logging 模块记录日志消息
    logging.info(label_string)


def calculate_execution_time(func):
    """
    计算函数的执行时间并打印
    :param func: 函数名
    :return:
    """

    def wrapper(*args, **kwargs):
        # 记录开始时间
        start_time = time.time()
        # 调用函数并获取结果
        result = func(*args, **kwargs)
        # 计算函数执行时间并打印
        log_message("Time elapsed for function {}: {:.2f} seconds".format(func.__name__, time.time() - start_time))
        return result

    return wrapper


def extract_email_address(text):
    """
    通过匹配'<'和'>'之间的字符来提取邮件地址。
    如果未找到匹配，则返回原始文本。
    :param text: 文本
    :return:
    """
    pattern = r'<(.+?)>'  # 匹配 '<' 和 '>' 中间的任何字符
    match = re.search(pattern, text)
    # 如果匹配到就返回'<'和'>'里面的内容，匹配不到直接返回原文
    if match:
        return match.group(1)
    else:
        return text


# 定义一个提取电子邮件地址的函数
def extract_text(email_address):
    """
    从电子邮件地址中提取地址，并返回地址
    :param email_address: 电子邮件地址
    :return:
    """
    # 使用 parseaddr 函数提取电子邮件地址的名称和地址
    # 并仅返回地址
    return parseaddr(email_address)[1]


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

current_directory_path = os.path.dirname(os.path.abspath(__file__))[:-3]


def create_directory(directory_name):
    """
    从主目录下创建你需要的目录。如果指定的目录不存在，则创建该目录。
    :param directory_name: 从主目录下延展的目录名
    :return:
    """

    directory_name = current_directory_path + directory_name
    # print(directory_name)
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
        log_message("成功创建目录：{}".format(directory_name))
    else:
        log_message("目录 {} 已存在".format(directory_name))


create_directory("features/subject_feature")
create_directory("features/fromto_feature")
create_directory("features/message_feature")


def lancaster_tokenizer(text):
    """
    将一个字符串分词并进行词干提取。
    :param text:文本
    :return:
    """
    # 初始化词干提取器
    lancaster_stemmer = LancasterStemmer()
    # 分词并将每个单词进行词干提取，并返回处理后的结果
    return [
        lancaster_stemmer.stem(token) for token in word_tokenize(text.lower())
    ]


# 获取停用词
stop_words = list(
    lancaster_tokenizer(
        " ".join(stopwords.words("english") + list(string.punctuation))))

# 导入分词器
tokenizer = lancaster_tokenizer

fg = "full"


@calculate_execution_time
def read_data(label="full"):
    """
    读取数据索引，并返回一个 Pandas DataFrame。
    :param label: 选择不同的数据集的标签
    :return:
    """
    global fg
    fg = label
    # 设置数据索引目录
    INDEX_DIR = DATA_DIR + "/" + label + "/index"
    # 读入数据索引
    index = pd.read_csv(INDEX_DIR,
                        header=None,  # 没有表头
                        names=["label", "index"],  # 设置列名
                        sep=" ")  # 使用空格作为分隔符
    # 获取文件索引
    # 使用 lambda 表达式和 apply 方法处理 'index' 列的数据
    index["index"] = index["index"].apply(lambda x: DATA_DIR + x[3:])
    return index


def clean_text(text):
    # 删除标点符号
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 删除停用词
    text_words = text.split()
    text_without_stopwords = [word for word in text_words if word.lower() not in stopwords.words("english")]
    text = " ".join(text_without_stopwords)
    # 去除乱码字符
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return text


def process_email(row):
    """
    一个处理电子邮件内容的函数，逐行处理并提取相关信息
    :param row: index的一行数据，包括文件路径和标签
    :return: 返回处理完成后的Series对象
    """
    error_list = []
    # 读取邮件内容并解码
    try:
        with open(row["index"], 'rb') as f:
            email_content = f.read()
            encoding = chardet.detect(email_content)['encoding']
            email_content = email_content.decode(encoding)
    except (UnicodeDecodeError, TypeError):
        error_list.append((row.name, row, "UnicodeDecodeError"))
        return None, error_list
    # 解析邮件内容
    parsed_email = email.message_from_string(email_content)
    # 初始化计数器和变量
    attachments = 0
    parts = 0
    html = 0
    email_body = ""
    # 遍历邮件的每个部分并提取相关信息
    if parsed_email.is_multipart():
        for part in parsed_email.walk():
            part_type = part.get_content_type()
            part_dispos = str(part.get("Content-Disposition"))
            parts += 1
            # 计算附件数量
            if "attachment" in part_dispos:
                attachments += 1
            # 检查是否有 HTML 部分
            if part_type == "text/html":
                html = 1
            # 提取纯文本正文
            if part_type == "text/plain" and "attachment" not in part_dispos:
                email_body = part.get_payload(decode=True)
    else:
        # 处理单部分邮件
        email_body = parsed_email.get_payload(decode=True)
        part_type = parsed_email.get_content_type()
        part_dispos = str(parsed_email.get("Content-Disposition"))
        # 计算附件数量和检查是否有 HTML 部分
        if "attachment" in part_dispos:
            attachments += 1
        if part_type == "text/html":
            html = 1
    # 确保邮件正文正确解码
    try:
        email_body = email_body.decode(
            errors='replace') if isinstance(email_body, bytes) else email_body
    except (UnicodeDecodeError, TypeError):
        error_list.append((row.name, row, "BodyDecodingError"))
        return None, error_list
    # 提取电子邮件的属性（如发件人、收件人、主题等）
    email_subject = parsed_email["Subject"]
    email_from = extract_text(parsed_email['From'])
    email_to = extract_text(parsed_email['To'])
    # 使用 BeautifulSoup 解析邮件正文并计算链接数量
    email_soup = BeautifulSoup(email_body, 'lxml')
    email_id_label = -7 if DATA_DIR[-8:-1] == 'trec06p' else (
        len(DATA_DIR) + 12 if DATA_DIR[-8:-1] == 'trec07p' else -7)
    email_id = row["index"][email_id_label:]
    email_label = 1 if row["label"] == "spam" else 0
    email_links = len(email_soup.find_all("a"))

    # 将提取到的电子邮件信息作为 pandas Series 对象返回
    return pd.Series([
        email_id, parts, attachments, html, email_from, email_to,
        email_subject, email_body, email_links, email_label
    ], index=columns), error_list


@calculate_execution_time
def parse_email_content(index):
    """
    处理电子邮件内容并将其转换为数据框，包括以下步骤：
        1. 批量处理电子邮件，并使用process_email函数对每封电子邮件进行处理
        2. 提取错误列表和数据框
        3. 删除处理过程中出现错误的行
        4. 将指定列转换为整数类型
        5. 存储结果
    :param index: index DataFrame数据
    :return: 异常列表
    """
    log_message("fg:" + fg)
    # 批量处理电子邮件
    processed_info = index.progress_apply(process_email, axis=1)
    # 提取错误列表和 DataFrame
    error_list = [x[1] for x in processed_info if x[0] is None]
    email_data = pd.concat([x[0] for x in processed_info if x[0] is not None],
                           axis=1).transpose()
    # 删除处理过程中出现错误的行
    email_data = email_data.dropna().reset_index(drop=True)
    email_data.dropna(inplace=True)
    log_message(email_data.shape)
    # 将指定列转换为整数类型
    email_data[["parts", "attachments", "html", "links", "spam"
                ]] = email_data[["parts", "attachments", "html", "links",
                                 "spam"]].astype("int32")
    email_data = email_data.dropna().reset_index(drop=True)
    log_message(email_data.shape)
    # 结果暂存
    email_data.to_csv(f"../data/email_{fg}.csv", index=False)

    email_data = pd.read_csv(f"../data/email_{fg}.csv")
    email_data = email_data.dropna().reset_index(drop=True)
    email_data.dropna(inplace=True)
    email_data.reset_index(drop=True)

    log_message(email_data.info())
    email_data.to_csv(f"../data/email_{fg}.csv", index=False)
    return error_list


def process_rows(df_chunk):
    return df_chunk.applymap(clean_text)


def clean_body_dask():
    log_message("fg:" + fg)
    log_message("处理时间大约40mins，请耐心等待。")
    cleaned_email_data = pd.read_csv(f"../data/email_{fg}.csv")
    log_message(cleaned_email_data.shape)
    # 转换为 Dask DataFrame
    dask_data = dd.from_pandas(cleaned_email_data, npartitions=cpu_count())
    # 使用 Dask 的 map_partitions 函数并行应用 clean_text 函数
    with ProgressBar():
        cleaned_data = dask_data.map_partitions(lambda df: df.assign(body=df['body'].apply(clean_text))).compute(
            scheduler='processes')
    # 将清洗过的数据保存到新文件

    cleaned_data.to_csv(f"../data/cleaned_email_{fg}.csv", index=False)

    cleaned_email_data = pd.read_csv(f"../data/cleaned_email_{fg}.csv")
    cleaned_email_data = cleaned_email_data.dropna().reset_index(drop=True)
    cleaned_email_data.dropna(inplace=True)
    cleaned_email_data.reset_index(drop=True)

    log_message(cleaned_email_data.info())
    cleaned_email_data.to_csv(f"../data/cleaned_email_{fg}.csv", index=False)


def process_column(data, column_name):
    """
    将指定列中的值转换为二进制值（0或1）。
    :param data: 数据
    :param column_name:指定列名
    :return:
    """
    data[column_name] = (data[column_name] > 0).astype(int)


@calculate_execution_time
def process_structural_info():
    """
    处理结构信息，然后保存
    :return:
    """
    # 读取CSV文件email_data
    email_data = pd.read_csv(f"../data/cleaned_email_{fg}.csv")
    # 根据email_data中包含的信息创建structural_info
    structural_info = pd.DataFrame({
        "has_multipart": email_data["parts"],  # 是否有多部分
        "has_html": email_data["html"],  # 是否有HTML内容
        "has_links": email_data["links"],  # 是否有链接
        "has_attachments": email_data["attachments"],  # 是否有附件
        "spam": email_data["spam"]  # 是否是垃圾邮件
    })
    # 处理structural_info的每一列
    process_column(structural_info, "has_multipart")
    process_column(structural_info, "has_links")
    process_column(structural_info, "has_attachments")
    # 输出structural_info的统计信息
    structural_info.describe()
    # 保存数据
    structural_info.to_csv(f"../data/email_structure_{fg}.csv", index=False)


def vectorize_and_save(data,
                       data_dir,
                       column,
                       feature=None,
                       vectorizer=None,
                       labels=None):
    """
    将数据的指定列向量化并保存
    :param data:需要向量化的数据
    :param data_dir:数据存储的目录
    :param column:需要向量化的数据列
    :param feature:向量化的方式，可以是count或tfidf
    :param vectorizer:可以提供一个已经创建的vectorizer，如果不提供，则会根据提供的特征类型创建一个
    :param labels:数据的标签
    :return:使用的vectorizer和向量化后的特征矩阵。
    """
    # 如果未提供vectorizer，则根据特征类型创建vectorizer
    if vectorizer is None:
        if feature == "count":
            vectorizer = CountVectorizer(max_df=0.8,
                                         min_df=5,
                                         max_features=10000,
                                         binary=False,
                                         ngram_range=(1, 2),
                                         tokenizer=tokenizer,  # 使用指定的tokenizer
                                         stop_words=stop_words)  # 停用词
        elif feature == "tfidf":
            vectorizer = TfidfVectorizer(max_df=0.8,
                                         min_df=5,
                                         max_features=10000,
                                         use_idf=True,
                                         smooth_idf=True,
                                         sublinear_tf=True,
                                         ngram_range=(1, 2),
                                         tokenizer=tokenizer,  # 使用指定的tokenizer
                                         stop_words=stop_words)  # 停用词
        else:
            # 如果提供的特征不是count或tfidf，则抛出错误
            raise ValueError(
                "Please provide a valid feature ('count' or 'tfidf') or a vectorizer."
            )

    # 记录正在提取特征的信息
    log_message("开始提取 {}_{} 特征。".format(column, feature))
    # 提取特征
    X = vectorizer.fit_transform(data[column])
    # 保存vectorizer
    pickle.dump(vectorizer, open(f"../features/{data_dir}/{column}_{feature}_vectorizer.p", "wb"))
    # 保存特征
    pickle.dump(X, open(f"../features/{data_dir}/{column}_{feature}_features.p", "wb"))
    # 记录已完成提取特征的信息
    log_message("完成提取 {}_{} 特征。".format(column, feature))

    # 如果提供了标签，则保存标签
    if labels is not None:
        pickle.dump(labels, open(f"../features/{data_dir}/labels.p", "wb"))

    # 打印特征矩阵的形状以及原始数据的形状
    log_message(X.shape)
    log_message(data[column].shape)
    return vectorizer, X


# 主题特征提取
@calculate_execution_time
def subject_feature_extraction(feature="count"):
    """
    主题特征提取
    :param feature: 向量化的方式，可以是count或tfidf
    :return:
    """
    # 读取邮件信息
    subject_content = pd.read_csv(f"../data/cleaned_email_{fg}.csv", usecols=["subject", "spam"]).fillna("")
    # 获取主题和spam列的内容，并将缺失值填充为空字符串

    if feature == "count":
        # Count Vectorizer
        # 调用vectorize_and_save函数提取主题的Count Vectorizer特征
        _, _ = vectorize_and_save(subject_content,
                                  "subject_feature",
                                  "subject",
                                  feature="count",
                                  labels=subject_content["spam"].values)
    elif feature == "tfidf":
        # Tfidf Vectorizer
        # 调用vectorize_and_save函数提取主题的Tfidf Vectorizer特征
        _, _ = vectorize_and_save(subject_content,
                                  "subject_feature",
                                  "subject",
                                  feature="tfidf",
                                  labels=subject_content["spam"].values)
    else:
        # 如果未提供支持的特征类型，则抛出错误
        raise ValueError("Sorry, no {} feature".format(feature))


# FromTo特征提取
@calculate_execution_time
def fromto_feature_extraction(feature="count"):
    """
    FromTo特征提取
    :param feature: 向量化的方式，可以是count或tfidf
    :return:
    """
    # 读取邮件信息
    fromto_content = pd.read_csv(f"../data/cleaned_email_{fg}.csv", usecols=["from", "to", "spam"]).fillna("")
    # 获取from、to和spam列的内容，并将缺失值填充为空字符串
    columns = ["from", "to"]

    for column in columns:
        if feature == "count":
            # Count Vectorizer
            # 调用vectorize_and_save函数提取from和to的Count Vectorizer特征
            _, _ = vectorize_and_save(fromto_content,
                                      "fromto_feature",
                                      column,
                                      feature="count",
                                      labels=fromto_content["spam"].values)
        elif feature == "tfidf":
            # Tf-idf Vectorizer
            # 调用vectorize_and_save函数提取from和to的Tf-idf Vectorizer特征
            _, _ = vectorize_and_save(fromto_content,
                                      "fromto_feature",
                                      column,
                                      feature="tfidf",
                                      labels=fromto_content["spam"].values)
        else:
            # 如果未提供支持的特征类型，则抛出错误
            raise ValueError("Sorry, no {} feature".format(feature))


# 提取message特征
@calculate_execution_time
def message_feature_extraction(feature="count"):
    """
    提取给定列的文本特征，并根据所选特征类型使用不同的向量化器。
    将提取的特征、向量化器和标签保存到pickle文件中。
    :param feature: 特征类型，默认为"count"。可选值为"count"、"tfidf"
    :return: 提取的文本特征矩阵
    """

    # 读取邮件信息，只选取所需列
    message_content = pd.read_csv(f"../data/cleaned_email_{fg}.csv", usecols=["body", "spam"]).fillna("")
    message_content.columns = ["message", "spam"]
    # 删除含有空值的行
    # email_data.dropna(axis=0, inplace=True)

    if feature == "count":
        # Count Vectorizer
        # 调用vectorize_and_save函数提取主题的Count Vectorizer特征
        _, _ = vectorize_and_save(message_content,
                                  "message_feature",
                                  "message",
                                  feature="count",
                                  labels=message_content["spam"].values)
    elif feature == "tfidf":
        # Tfidf Vectorizer
        # 调用vectorize_and_save函数提取主题的Tfidf Vectorizer特征
        _, _ = vectorize_and_save(message_content,
                                  "message_feature",
                                  "message",
                                  feature="tfidf",
                                  labels=message_content["spam"].values)
    else:
        # 如果未提供支持的特征类型，则抛出错误
        raise ValueError("Sorry, no {} feature".format(feature))


# data_process run
def run(folder_path, word_type):
    global DATA_DIR
    DATA_DIR = folder_path

    log_message(f"文件夹路径: {folder_path}")
    log_message(f"数据集类型: {word_type}")

    index = read_data(word_type)
    # parse_email_content(index)
    # process_structural_info()

    # subject_feature_extraction(feature="count")
    # fromto_feature_extraction(feature="count")
    # extract_message_feature(feature="count")


def process_trec06p(fgg="full"):
    # 设置目录
    global DATA_DIR
    DATA_DIR = "D:/py/pythonProjectSpam/trec06p/"
    # 读取并处理数据
    index = read_data(fgg)
    print(index)
    parse_email_content(index)
    clean_body_dask()
    # 查看数据信息
    data = pd.read_csv(f"../data/cleaned_email_{fgg}.csv")
    print(data.head())
    print(data.shape)
    print(data.info())


def process_trec07p(fgg="full"):
    # 设置目录
    global DATA_DIR
    DATA_DIR = "D:/py/pythonProjectSpam/trec07p/"
    # 读取并处理数据
    # index = read_data(fgg)
    # print(index)
    global fg
    fg = "07_" + fgg
    # parse_email_content(index)
    clean_body_dask()
    # 查看数据信息
    data = pd.read_csv(f"../data/cleaned_email_{fg}.csv")
    print(data.head())
    print(data.shape)
    print(data.info())


if __name__ == "__main__":
    fg = "full"

    # process_trec06p(fg)

    process_trec07p(fg)

    # process_structural_info()
    #
    # subject_feature_extraction(feature="count")
    # fromto_feature_extraction(feature="count")
    # message_feature_extraction(feature="count")
    #
    # subject_feature_extraction(feature="tfidf")
    # fromto_feature_extraction(feature="tfidf")
    # message_feature_extraction(feature="tfidf")
