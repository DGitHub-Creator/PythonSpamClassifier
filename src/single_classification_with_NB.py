#!/usr/bin/env python
# coding: utf-8

#  Copyright (c) 2023 DZX.
#
#  All rights reserved.
#
#  This software is protected by copyright law and international treaties. No part of this software may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the copyright owner.
#
#  For permission requests, please contact the copyright owner at the address below.
#
#  DZX
#
#  xindemicro@outlook.com
#

import datetime
import email
import logging
import os
import pickle
import re
import string
import time
import warnings

# 导入相关库
import chardet
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from keras.utils import pad_sequences
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix, hstack
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

warnings.filterwarnings("ignore")

project_directory_path = os.path.dirname(os.path.abspath(__file__))[:-3]
# 配置日志
logging.basicConfig(filename=project_directory_path + "logs/single_run_log.txt",
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
        log_message("[->]函数 {} 的执行时间为 {:.2f} 秒".format(func.__name__, time.time() - start_time))
        # log_message("Time elapsed for function {}: {:.2f} seconds".format(func.__name__, time.time() - start_time))
        return result

    return wrapper


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


def stem_tokenizer(text):
    """
        将一个字符串分词并进行词干提取。
        :param text:文本
        :return:
        """
    # 初始化词干提取器
    porter_stemmer = PorterStemmer()

    stop_words = list(stopwords.words("english"))
    words = [porter_stemmer.stem(token) for token in word_tokenize(text.lower())]
    return " ".join([w for w in words if w not in stop_words])


# 获取停用词
stop_words = list(
    lancaster_tokenizer(
        " ".join(stopwords.words("english") + list(string.punctuation))))

# 导入分词器
tokenizer = lancaster_tokenizer


# 函数：检测文件编码
def detect_encoding(file):
    with open(file, 'rb') as f:
        result = chardet.detect(f.read())  # 使用chardet库检测文件编码
    return result['encoding']  # 返回文件编码


# 函数：从文本中提取尖括号内的内容
def extract_text(text):
    if text is None:  # 如果输入为None，返回空字符串
        return ''
    match = re.search(r'<(.+?)>', text)  # 正则表达式匹配尖括号内的内容
    return match.group(1) if match else text  # 如果匹配到内容则返回，否则返回原文本


# 函数：解析电子邮件内容
@calculate_execution_time
def parse_email_from_file(email_path):
    # 初始化一个空的DataFrame，用于存储邮件信息
    email_data = pd.DataFrame({},
                              columns=[
                                  "email_id", "parts", "attachments", "html",
                                  "from", "to", "subject", "body", "links"
                              ])
    # 尝试读取邮件文件
    try:
        encoding = detect_encoding(email_path)  # 检测邮件文件编码
        with open(email_path, 'r', encoding=encoding) as f:
            email_content = f.read()  # 读取邮件内容
    except (FileNotFoundError, UnicodeDecodeError, Exception) as e:
        print("读取电子邮件文件时发生错误：", e)
        return email_data  # 如果读取文件时发生错误，则返回空的DataFrame
    # 将邮件内容解析为电子邮件消息对象
    parsed_email = email.message_from_string(email_content)
    # 初始化计数器：附件数量、部分数量、是否包含HTML正文
    attachments = parts = html = 0
    # 初始化电子邮件正文为空字节
    email_body = b''
    # 遍历邮件的所有部分
    for part in parsed_email.walk():
        part_type = part.get_content_type()  # 获取部分类型
        part_dispos = str(part.get("Content-Disposition"))  # 获取部分描述信息
        parts += 1  # 部分计数器+1
        attachments += "attachment" in part_dispos  # 如果部分是附件，则附件计数器+1
        html |= part_type == "text/html"  # 如果部分是HTML正文，则html计数器置为1
        # 如果找到纯文本正文且不是附件，则提取正文内容并跳出循环
        if part_type == "text/plain" and "attachment" not in part_dispos:
            email_body = part.get_payload(decode=True)
            break
    # 尝试提取邮件主题、发件人和收件人
    try:
        email_subject = parsed_email["Subject"]
        email_from = extract_text(parsed_email.get('From',
                                                   ''))  # 使用get方法并提供默认值为空字符串
        email_to = extract_text(parsed_email.get('To',
                                                 ''))  # 使用get方法并提供默认值为空字符串
        # 如果发件人或收件人为空字符串，引发异常
        if not email_from:
            email_from = b''
        if not email_to:
            email_to = b''
    except (KeyError, ValueError) as e:
        print("解析电子邮件时发生错误：", e)
        return email_data  # 如果发生错误，则返回空的DataFrame
    # 使用BeautifulSoup解析器处理电子邮件正文
    email_soup = BeautifulSoup(email_body, features="lxml")
    # 从文件名中提取电子邮件的唯一标识符（文件名的最后7位）
    email_id = email_path[-7:]
    # 统计邮件正文中的链接数量
    email_links = len(email_soup.find_all("a"))
    if email_id is None:
        email_id = b''
    if parts is None:
        parts = b''
    if attachments is None:
        attachments = b''
    if html is None:
        html = b''
    if email_from is None:
        email_from = b''
    if email_to is None:
        email_to = b''
    if email_subject is None:
        email_subject = b''
    if email_body is None:
        email_body = b''
    if email_links is None:
        email_links = b''
    # 将提取到的邮件相关信息存储到新的DataFrame中
    info = pd.DataFrame([[
        email_id, parts, attachments, html, email_from, email_to,
        email_subject, email_body, email_links
    ]],
        columns=email_data.columns)
    # 将新的邮件信息添加到之前初始化的空的DataFrame中
    email_data = pd.concat([email_data, info], axis=0, ignore_index=True)
    # 将指定列转换为整数类型
    email_data[["parts", "attachments", "html", "links"
                ]] = email_data[["parts", "attachments", "html", "links"]].astype("int32")
    email_data = email_data.astype(
        {'email_id': 'string', 'from': 'string', 'to': 'string', 'subject': 'string', 'body': 'string'})
    # 打印邮件信息
    # log_message(email_data)
    # 返回包含邮件信息的DataFrame
    return email_data


# 函数：解析电子邮件内容
@calculate_execution_time
def parse_email_from_content(email_content):
    # 初始化一个空的DataFrame，用于存储邮件信息
    email_data = pd.DataFrame({},
                              columns=[
                                  "email_id", "parts", "attachments", "html",
                                  "from", "to", "subject", "body", "links"
                              ])
    # 将邮件内容解析为电子邮件消息对象
    parsed_email = email.message_from_string(email_content)
    # 初始化计数器：附件数量、部分数量、是否包含HTML正文
    attachments = parts = html = 0
    # 初始化电子邮件正文为空字节
    email_body = b''
    # 遍历邮件的所有部分
    for part in parsed_email.walk():
        part_type = part.get_content_type()  # 获取部分类型
        part_dispos = str(part.get("Content-Disposition"))  # 获取部分描述信息
        parts += 1  # 部分计数器+1
        attachments += "attachment" in part_dispos  # 如果部分是附件，则附件计数器+1
        html |= part_type == "text/html"  # 如果部分是HTML正文，则html计数器置为1
        # 如果找到纯文本正文且不是附件，则提取正文内容并跳出循环
        if part_type == "text/plain" and "attachment" not in part_dispos:
            email_body = part.get_payload(decode=True)
            break
    # 尝试提取邮件主题、发件人和收件人
    try:
        email_subject = parsed_email["Subject"]
        email_from = extract_text(parsed_email.get('From',
                                                   ''))  # 使用get方法并提供默认值为空字符串
        email_to = extract_text(parsed_email.get('To',
                                                 ''))  # 使用get方法并提供默认值为空字符串
        # 如果发件人或收件人为空字符串，引发异常
        if not email_from or not email_to:
            raise ValueError("发件人或收件人为空")
    except (KeyError, ValueError) as e:
        print("解析电子邮件时发生错误：", e)
        return email_data  # 如果发生错误，则返回空的DataFrame
    # 使用BeautifulSoup解析器处理电子邮件正文
    email_soup = BeautifulSoup(email_body, features="lxml")
    # 从文件名中提取电子邮件的唯一标识符（文件名的最后7位）
    email_id = "user_input"
    # 统计邮件正文中的链接数量
    email_links = len(email_soup.find_all("a"))
    if email_id is None:
        email_id = b''
    if parts is None:
        parts = b''
    if attachments is None:
        attachments = b''
    if html is None:
        html = b''
    if email_from is None:
        email_from = b''
    if email_to is None:
        email_to = b''
    if email_subject is None:
        email_subject = b''
    if email_body is None:
        email_body = b''
    if email_links is None:
        email_links = b''
    # 将提取到的邮件相关信息存储到新的DataFrame中
    info = pd.DataFrame([[
        email_id, parts, attachments, html, email_from, email_to,
        email_subject, email_body, email_links
    ]],
        columns=email_data.columns)
    # 将新的邮件信息添加到之前初始化的空的DataFrame中
    email_data = pd.concat([email_data, info], axis=0, ignore_index=True)
    # 将指定列转换为整数类型
    email_data[["parts", "attachments", "html", "links"
                ]] = email_data[["parts", "attachments", "html", "links"]].astype("int32")
    email_data = email_data.astype(
        {'email_id': 'string', 'from': 'string', 'to': 'string', 'subject': 'string', 'body': 'string'})
    # 打印邮件信息
    # log_message(email_data)
    # 返回包含邮件信息的DataFrame
    return email_data


@calculate_execution_time
def process_structure_info(email_data):
    # 根据email_data中包含的信息创建structural_info
    structural_info = pd.DataFrame({
        "multipart": email_data["parts"],
        "html": email_data["html"],
        "links": email_data["links"],
        "attachments": email_data["attachments"]
    })
    # 对multipart列进行处理，将值大于0的置为1，其余置为0
    structural_info["multipart"] = (structural_info["multipart"] >
                                    0).astype(int)
    # 对html列进行处理，转为int
    structural_info["html"] = structural_info["html"].astype(int)
    # 对links列进行处理，将值大于0的置为1，其余置为0
    structural_info["links"] = (structural_info["links"] > 0).astype(int)
    # 对attachments列进行处理，将值大于0的置为1，其余置为0
    structural_info["attachments"] = (structural_info["attachments"] >
                                      0).astype(int)
    # 返回structural_info
    return structural_info


@calculate_execution_time
def process_subject_feature(email_data):
    # 从邮件信息中提取主题和是否为垃圾邮件的信息，填充缺失值为""
    subject_content = email_data[["subject"]]
    subject_content.fillna(value="", inplace=True)
    subject_count_Vectorizer = pickle.load(
        open(project_directory_path + "features/subject_feature/subject_count_vectorizer.p", "rb"))
    X_count = subject_count_Vectorizer.transform(subject_content["subject"])
    return X_count


@calculate_execution_time
def process_fromto_feature(email_data):
    fromto_content = email_data[["from", "to"]]
    fromto_content.fillna(value="", inplace=True)
    from_count_Vectorizer = pickle.load(
        open(project_directory_path + "features/fromto_feature/from_count_vectorizer.p", "rb"))
    to_count_Vectorizer = pickle.load(
        open(project_directory_path + "features/fromto_feature/to_count_vectorizer.p", "rb"))
    # 对fromto_content的主题进行特征提取，并转换成矩阵
    from_X_count = from_count_Vectorizer.transform(fromto_content["from"])
    to_X_count = to_count_Vectorizer.transform(fromto_content["to"])
    return from_X_count, to_X_count


@calculate_execution_time
def process_message_body(email_data):
    # 从邮件信息中选取正文和是否垃圾邮件标签列
    email_body = email_data[["body"]]
    # 删除含有空值的行
    email_body.fillna(value="", inplace=True)
    message_count_Vectorizer = pickle.load(
        open(project_directory_path + "features/message_feature/message_count_vectorizer.p", "rb"))
    message_X_count = message_count_Vectorizer.transform(email_body["body"].iloc[:])
    return message_X_count


@calculate_execution_time
def run_file_with_NB(path, feature="count"):
    single_email_data = parse_email_from_file(path)
    # structure_info
    structure_info = process_structure_info(single_email_data)
    # structure_info = structure_info.set_index('multipart', drop=True)
    structure_info = csr_matrix(structure_info)
    # subject_feature
    subject_feature = process_subject_feature(single_email_data)
    # fromto_feature
    from_feature, to_feature = process_fromto_feature(single_email_data)
    # message_body
    message_body = process_message_body(single_email_data)
    # log_message(structure_info.shape, subject_feature.shape, from_feature.shape, to_feature.shape, message_body.shape)
    # Combine features together
    X_set = csr_matrix(
        hstack([
            message_body, from_feature, to_feature, subject_feature,
            structure_info
        ]))
    with open(project_directory_path + "models/CombinedFeatNB_{}.p".format(feature), "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_set)
    # log_message(y_pred)
    log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
    return True if y_pred else False


@calculate_execution_time
def run_content_with_NB(email_content, feature="count"):
    single_email_data = parse_email_from_content(email_content)
    # structure_info
    structure_info = process_structure_info(single_email_data)
    # structure_info = structure_info.set_index('multipart', drop=True)
    structure_info = csr_matrix(structure_info)
    # subject_feature
    subject_feature = process_subject_feature(single_email_data)
    # fromto_feature
    from_feature, to_feature = process_fromto_feature(single_email_data)
    # message_body
    message_body = process_message_body(single_email_data)
    # log_message(structure_info.shape, subject_feature.shape, from_feature.shape, to_feature.shape, message_body.shape)
    # Combine features together
    X_set = csr_matrix(
        hstack([
            message_body, from_feature, to_feature, subject_feature,
            structure_info
        ]))
    with open(project_directory_path + "models/CombinedFeatNB_{}.p".format(feature), "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_set)
    # log_message(y_pred)
    log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
    return True if y_pred else False


@calculate_execution_time
def run_file_with_LSTM(path):
    max_vocab = 600000
    max_len = 2000
    # 读取邮件内容
    single_email_data = parse_email_from_file(path)

    single_email_data['combined'] = single_email_data['from'] + ' ' + single_email_data['to'] + ' ' + single_email_data[
        'body']

    messages = []
    for text in tqdm(single_email_data['combined']):
        messages.append(stem_tokenizer(str(text)))

    # 读取 tokenizer
    with open(project_directory_path + '/data/vectokenizer_body.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenizer.fit_on_texts(messages)
    sequences = tokenizer.texts_to_sequences(messages)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=max_len)

    name = "message_glove"
    model_file = project_directory_path + "/models/LSTM_model_" + name
    # 从指定路径加载模型
    model = load_model(model_file)

    y_pred = model.predict(data)
    y_pred = np.round(np.squeeze(y_pred)).astype(int)

    log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
    return True if y_pred else False


@calculate_execution_time
def run_content_with_LSTM(email_content):
    max_vocab = 600000
    max_len = 2000
    # 读取邮件内容
    single_email_data = parse_email_from_content(email_content)

    single_email_data['combined'] = single_email_data['from'] + ' ' + single_email_data['to'] + ' ' + single_email_data[
        'body']

    messages = []
    for text in tqdm(single_email_data['combined']):
        messages.append(stem_tokenizer(str(text)))

    # 读取 tokenizer
    with open(project_directory_path + '/data/vectokenizer_body.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    tokenizer.fit_on_texts(messages)
    sequences = tokenizer.texts_to_sequences(messages)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=max_len)

    name = "message_glove"
    model_file = project_directory_path + "/models/LSTM_model_" + name
    # 从指定路径加载模型
    model = load_model(model_file)

    y_pred = model.predict(data)
    y_pred = np.round(np.squeeze(y_pred)).astype(int)

    log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
    return True if y_pred else False


@calculate_execution_time
def run_file(file_path, model_name):
    log_message(f"[!]{model_name}运行开始]")
    with open(file_path, 'r', encoding='utf-8') as f:
        file_contents = f.read()
    label = None
    if model_name == "朴素贝叶斯":
        label = run_file_with_NB(file_path)
    elif model_name == "LSTM":
        label = run_file_with_LSTM(file_path)
        pass
    else:
        log_message(f"没有{model_name}模型！")
        return file_contents, label
    log_message(f"[!]{model_name}运行结束")
    return file_contents, label


@calculate_execution_time
def run_content(email_content, model_name):
    log_message(f"[!]{model_name}运行开始]")
    label = None
    if model_name == "朴素贝叶斯":
        label = run_content_with_NB(email_content)
    elif model_name == "LSTM":
        label = run_content_with_LSTM(email_content)
        pass
    else:
        log_message(f"没有{model_name}模型！")
        return email_content, label
    log_message(f"[!]{model_name}运行结束")
    return email_content, label

# if __name__ == "__main__":
#     run_file_with_NB(r"D:\py\pythonProjectSpam\trec06p\data\000\000", feature="tfidf")
