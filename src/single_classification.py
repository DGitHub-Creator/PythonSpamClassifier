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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

    try:
        # 使用 logging 模块记录日志消息
        logging.info(label_string)
    except Exception as e:
        # 发生错误时，将错误信息记录到日志文件中
        logging.error("Failed to log message: {}".format(e))


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


def detect_encoding(file):
    try:
        with open(file, 'rb') as f:
            result = chardet.detect(f.read())
    except IOError as e:
        print("发生IOError错误: {}".format(e))
        logging.error("发生IOError错误: {}".format(e))
    return result['encoding']  # 返回文件编码


def extract_text(text):
    if text is None:  # 如果输入为None，返回空字符串
        return ''
    match = re.search(r'<(.+?)>', text)  # 正则表达式匹配尖括号内的内容
    return match.group(1) if match else text  # 如果匹配到内容则返回，否则返回原文本


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
        print("读取电子邮件文件时发生错误: {}".format(e))
        logging.error("读取电子邮件文件时发生错误: {}".format(e))
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
    def binary_transform(col):
        return (col > 0).astype(int)

    structural_info = pd.DataFrame({
        "multipart": binary_transform(email_data["parts"]),
        "html": email_data["html"].astype(int),
        "links": binary_transform(email_data["links"]),
        "attachments": binary_transform(email_data["attachments"])
    })

    return structural_info


@calculate_execution_time
def load_pickle_file(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except (IOError, pickle.PickleError) as e:
        print(f"发生错误: {e}")
        logging.error(f"发生错误: {e}")
        return None


@calculate_execution_time
def process_subject_feature(email_data):
    subject_content = email_data[["subject"]].fillna(value="")
    subject_count_Vectorizer = load_pickle_file(
        project_directory_path + "features/subject_feature/subject_count_vectorizer.p")

    if subject_count_Vectorizer:
        return subject_count_Vectorizer.transform(subject_content["subject"])
    return None


@calculate_execution_time
def process_fromto_feature(email_data):
    fromto_content = email_data[["from", "to"]].fillna(value="")
    from_count_Vectorizer = load_pickle_file(project_directory_path + "features/fromto_feature/from_count_vectorizer.p")
    to_count_Vectorizer = load_pickle_file(project_directory_path + "features/fromto_feature/to_count_vectorizer.p")

    if from_count_Vectorizer and to_count_Vectorizer:
        from_X_count = from_count_Vectorizer.transform(fromto_content["from"])
        to_X_count = to_count_Vectorizer.transform(fromto_content["to"])
        return from_X_count, to_X_count
    return None, None


@calculate_execution_time
def process_message_body(email_data):
    email_body = email_data[["body"]].fillna(value="")
    message_count_Vectorizer = load_pickle_file(
        project_directory_path + "features/message_feature/message_count_vectorizer.p")

    if message_count_Vectorizer:
        return message_count_Vectorizer.transform(email_body["body"].iloc[:])
    return None


@calculate_execution_time
def process_email_data(email_data, model_name, feature=None):
    if model_name == "Naive Bayes":
        structure_info = process_structure_info(email_data)
        structure_info = csr_matrix(structure_info)
        subject_feature = process_subject_feature(email_data)
        from_feature, to_feature = process_fromto_feature(email_data)
        message_body = process_message_body(email_data)
        X_set = csr_matrix(
            hstack([
                message_body, from_feature, to_feature, subject_feature,
                structure_info
            ]))
        try:
            with open(project_directory_path + f"models/CombinedFeatNB_{feature}.p", "rb") as f:
                model = pickle.load(f)
        except IOError as e:
            # 记录错误日志
            print("发生IOError错误: {}".format(e))
            logging.error("发生IOError错误: {}".format(e))
        except pickle.PickleError as e:
            # 记录错误日志
            print("发生PickleError错误: {}".format(e))
            logging.error("发生PickleError错误: {}".format(e))
    elif model_name in ("LSTM", "Transformer"):
        max_vocab = 600000
        max_len = 2000 if model_name == "LSTM" else 1000
        email_data['combined'] = email_data['from'] + ' ' + email_data['to'] + ' ' + email_data['body']
        messages = [stem_tokenizer(str(text)) for text in tqdm(email_data['combined'])]

        tokenizer_path = project_directory_path + f"/data/vectokenizer{'_max_len=1000' if model_name == 'Transformer' else ''}.pickle"
        try:
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
        except IOError as e:
            # 记录错误日志
            print("发生IOError错误: {}".format(e))
            logging.error("发生IOError错误: {}".format(e))
        except pickle.PickleError as e:
            # 记录错误日志
            print("发生PickleError错误: {}".format(e))
            logging.error("发生PickleError错误: {}".format(e))
        sequences = tokenizer.texts_to_sequences(messages)
        data = pad_sequences(sequences, maxlen=max_len)
        X_set = data

        model_file = project_directory_path + f"/models/{model_name}_model_message_glove"
        model = load_model(model_file)
    else:
        raise ValueError("Invalid model_name")

    y_pred = model.predict(X_set)
    y_pred = np.round(np.squeeze(y_pred)).astype(int)
    log_message(f"[Predict result] : {y_pred} " + ("This is a spam email." if y_pred else "This is not a spam email."))
    return True if y_pred else False


@calculate_execution_time
def run_file_with_model(file_path, model_name, feature=None):
    email_data = parse_email_from_file(file_path)
    return email_data["body"][0], process_email_data(email_data, model_name, feature)


@calculate_execution_time
def run_content_with_model(email_content, model_name, feature=None):
    email_data = parse_email_from_content(email_content)
    return email_data["body"][0], process_email_data(email_data, model_name, feature)


@calculate_execution_time
def run_file(file_path, model_name):
    log_message(f"[!]{model_name}运行开始]")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_contents = f.read()
    except IOError as e:
        # 记录错误日志
        print("发生IOError错误: {}".format(e))
        logging.error("发生IOError错误: {}".format(e))
    except UnicodeDecodeError as e:
        # 记录错误日志
        print("发生UnicodeDecodeError错误: {}".format(e))
        logging.error("发生UnicodeDecodeError错误: {}".format(e))
    label = None
    if model_name == "Naive Bayes":
        file_contents, label = run_file_with_model(file_path, model_name, feature="tfidf")
    elif model_name == "LSTM":
        file_contents, label = run_file_with_model(file_path, model_name, feature=None)
    elif model_name == "Transformer":
        file_contents, label = run_file_with_model(file_path, model_name, feature=None)
    else:
        log_message(f"没有{model_name}模型！")
        return file_contents, label
    log_message(f"[!]{model_name}运行结束")
    return file_contents, label


@calculate_execution_time
def run_content(email_content, model_name):
    log_message(f"[!]{model_name}运行开始]")
    label = None
    if model_name == "Naive Bayes":
        email_content, label = run_content_with_model(email_content, model_name, feature="tfidf")
    elif model_name == "LSTM":
        email_content, label = run_content_with_model(email_content, model_name, feature=None)
    elif model_name == "Transformer":
        email_content, label = run_content_with_model(email_content, model_name, feature=None)
    else:
        log_message(f"没有{model_name}模型！")
        return email_content, label
    log_message(f"[!]{model_name}运行结束")
    return email_content, label

# if __name__ == "__main__":
# run_file_with_NB(r"D:\py\pythonProjectSpam\trec06p\data\000\000", feature="tfidf")
# run_file_with_Transformer(r"D:\py\pythonProjectSpam\trec06p\data\000\000")
