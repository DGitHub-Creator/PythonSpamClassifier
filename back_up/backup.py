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

#
#
# @Time    : 2023/5/3 16:11
# @Author  : DZX
# @Email   : xindemicro@outlook.com
# @File    : backup.py
# @Software: PyCharm







# @calculate_execution_time
# def process_structure_info(email_data):
#     # 根据email_data中包含的信息创建structural_info
#     structural_info = pd.DataFrame({
#         "multipart": email_data["parts"],
#         "html": email_data["html"],
#         "links": email_data["links"],
#         "attachments": email_data["attachments"]
#     })
#     # 对multipart列进行处理，将值大于0的置为1，其余置为0
#     structural_info["multipart"] = (structural_info["multipart"] >
#                                     0).astype(int)
#     # 对html列进行处理，转为int
#     structural_info["html"] = structural_info["html"].astype(int)
#     # 对links列进行处理，将值大于0的置为1，其余置为0
#     structural_info["links"] = (structural_info["links"] > 0).astype(int)
#     # 对attachments列进行处理，将值大于0的置为1，其余置为0
#     structural_info["attachments"] = (structural_info["attachments"] >
#                                       0).astype(int)
#     # 返回structural_info
#     return structural_info
#
#
# @calculate_execution_time
# def process_subject_feature(email_data):
#     X_count = None
#     # 从邮件信息中提取主题和是否为垃圾邮件的信息，填充缺失值为""
#     subject_content = email_data[["subject"]]
#     subject_content.fillna(value="", inplace=True)
#     try:
#         subject_count_Vectorizer = pickle.load(
#             open(project_directory_path + "features/subject_feature/subject_count_vectorizer.p", "rb"))
#         X_count = subject_count_Vectorizer.transform(subject_content["subject"])
#     except IOError as e:
#         # 记录错误日志
#         print("发生IOError错误: {}".format(e))
#         logging.error("发生IOError错误: {}".format(e))
#     except pickle.PickleError as e:
#         # 记录错误日志
#         print("发生PickleError错误: {}".format(e))
#         logging.error("发生PickleError错误: {}".format(e))
#     return X_count
#
#
# @calculate_execution_time
# def process_fromto_feature(email_data):
#     fromto_content = email_data[["from", "to"]]
#     fromto_content.fillna(value="", inplace=True)
#     from_count_Vectorizer = pickle.load(
#         open(project_directory_path + "features/fromto_feature/from_count_vectorizer.p", "rb"))
#     to_count_Vectorizer = pickle.load(
#         open(project_directory_path + "features/fromto_feature/to_count_vectorizer.p", "rb"))
#     # 对fromto_content的主题进行特征提取，并转换成矩阵
#     from_X_count = from_count_Vectorizer.transform(fromto_content["from"])
#     to_X_count = to_count_Vectorizer.transform(fromto_content["to"])
#     return from_X_count, to_X_count
#
#
# @calculate_execution_time
# def process_message_body(email_data):
#     # 从邮件信息中选取正文和是否垃圾邮件标签列
#     email_body = email_data[["body"]]
#     # 删除含有空值的行
#     email_body.fillna(value="", inplace=True)
#     message_count_Vectorizer = pickle.load(
#         open(project_directory_path + "features/message_feature/message_count_vectorizer.p", "rb"))
#     message_X_count = message_count_Vectorizer.transform(email_body["body"].iloc[:])
#     return message_X_count



# @calculate_execution_time
# def run_file_with_NB(file_path, feature="tfidf"):
#     single_email_data = parse_email_from_file(file_path)
#     # structure_info
#     structure_info = process_structure_info(single_email_data)
#     # structure_info = structure_info.set_index('multipart', drop=True)
#     structure_info = csr_matrix(structure_info)
#     # subject_feature
#     subject_feature = process_subject_feature(single_email_data)
#     # fromto_feature
#     from_feature, to_feature = process_fromto_feature(single_email_data)
#     # message_body
#     message_body = process_message_body(single_email_data)
#     # log_message(structure_info.shape, subject_feature.shape, from_feature.shape, to_feature.shape, message_body.shape)
#     # Combine features together
#     X_set = csr_matrix(
#         hstack([
#             message_body, from_feature, to_feature, subject_feature,
#             structure_info
#         ]))
#     with open(project_directory_path + "models/CombinedFeatNB_{}.p".format(feature), "rb") as f:
#         model = pickle.load(f)
#     y_pred = model.predict(X_set)
#     # log_message(y_pred)
#     log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
#     return True if y_pred else False
#
#
# @calculate_execution_time
# def run_content_with_NB(email_content, feature="tfidf"):
#     single_email_data = parse_email_from_content(email_content)
#     # structure_info
#     structure_info = process_structure_info(single_email_data)
#     # structure_info = structure_info.set_index('multipart', drop=True)
#     structure_info = csr_matrix(structure_info)
#     # subject_feature
#     subject_feature = process_subject_feature(single_email_data)
#     # fromto_feature
#     from_feature, to_feature = process_fromto_feature(single_email_data)
#     # message_body
#     message_body = process_message_body(single_email_data)
#     # log_message(structure_info.shape, subject_feature.shape, from_feature.shape, to_feature.shape, message_body.shape)
#     # Combine features together
#     X_set = csr_matrix(
#         hstack([
#             message_body, from_feature, to_feature, subject_feature,
#             structure_info
#         ]))
#     with open(project_directory_path + "models/CombinedFeatNB_{}.p".format(feature), "rb") as f:
#         model = pickle.load(f)
#     y_pred = model.predict(X_set)
#     # log_message(y_pred)
#     log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
#     return True if y_pred else False
#
#
# @calculate_execution_time
# def run_file_with_LSTM(file_path):
#     max_vocab = 600000
#     max_len = 2000
#     # 读取邮件内容
#     single_email_data = parse_email_from_file(file_path)
#
#     single_email_data['combined'] = single_email_data['from'] + ' ' + single_email_data['to'] + ' ' + single_email_data[
#         'body']
#
#     messages = []
#     for text in tqdm(single_email_data['combined']):
#         messages.append(stem_tokenizer(str(text)))
#
#     # 读取 tokenizer
#     with open(project_directory_path + '/data/vectokenizer.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#
#     # tokenizer.fit_on_texts(messages)
#     sequences = tokenizer.texts_to_sequences(messages)
#     word_index = tokenizer.word_index
#     data = pad_sequences(sequences, maxlen=max_len)
#
#     name = "message_glove"
#     model_file = project_directory_path + "/models/LSTM_model_" + name
#     # 从指定路径加载模型
#     model = load_model(model_file)
#
#     y_pred = model.predict(data)
#     y_pred = np.round(np.squeeze(y_pred)).astype(int)
#
#     log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
#     return True if y_pred else False
#
#
# @calculate_execution_time
# def run_content_with_LSTM(email_content):
#     max_vocab = 600000
#     max_len = 2000
#     # 读取邮件内容
#     single_email_data = parse_email_from_content(email_content)
#
#     single_email_data['combined'] = single_email_data['from'] + ' ' + single_email_data['to'] + ' ' + single_email_data[
#         'body']
#
#     messages = []
#     for text in tqdm(single_email_data['combined']):
#         messages.append(stem_tokenizer(str(text)))
#
#     # 读取 tokenizer
#     with open(project_directory_path + '/data/vectokenizer.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#
#     # tokenizer.fit_on_texts(messages)
#     sequences = tokenizer.texts_to_sequences(messages)
#     word_index = tokenizer.word_index
#     data = pad_sequences(sequences, maxlen=max_len)
#
#     name = "message_glove"
#     model_file = project_directory_path + "/models/LSTM_model_" + name
#     # 从指定路径加载模型
#     model = load_model(model_file)
#
#     y_pred = model.predict(data)
#     y_pred = np.round(np.squeeze(y_pred)).astype(int)
#
#     log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
#     return True if y_pred else False
#
# calculate_execution_time
# def run_file_with_Transformer(file_path):
#     max_vocab = 600000
#     max_len = 1000
#     # 读取邮件内容
#     single_email_data = parse_email_from_file(file_path)
#
#     single_email_data['combined'] = single_email_data['from'] + ' ' + single_email_data['to'] + ' ' + single_email_data[
#         'body']
#
#     messages = []
#     for text in tqdm(single_email_data['combined']):
#         messages.append(stem_tokenizer(str(text)))
#
#     # 读取 tokenizer
#     with open(project_directory_path + '/data/vectokenizer_max_len=1000.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#
#     # tokenizer.fit_on_texts(messages)
#     sequences = tokenizer.texts_to_sequences(messages)
#     word_index = tokenizer.word_index
#     data = pad_sequences(sequences, maxlen=max_len)
#
#     name = "message_glove"
#     model_file = project_directory_path + "/models/Transformer_model_" + name
#     # 从指定路径加载模型
#     model = load_model(model_file)
#
#     y_pred = model.predict(data)
#     y_pred = np.round(np.squeeze(y_pred)).astype(int)
#
#     log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
#     return True if y_pred else False
#
# calculate_execution_time
# def run_content_with_Transformer(email_content):
#     max_vocab = 600000
#     max_len = 1000
#     # 读取邮件内容
#     single_email_data = parse_email_from_content(email_content)
#
#     single_email_data['combined'] = single_email_data['from'] + ' ' + single_email_data['to'] + ' ' + single_email_data[
#         'body']
#
#     messages = []
#     for text in tqdm(single_email_data['combined']):
#         messages.append(stem_tokenizer(str(text)))
#
#     # 读取 tokenizer
#     with open(project_directory_path + '/data/vectokenizer_max_len=1000.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
#
#     # tokenizer.fit_on_texts(messages)
#     sequences = tokenizer.texts_to_sequences(messages)
#     word_index = tokenizer.word_index
#     data = pad_sequences(sequences, maxlen=max_len)
#
#     name = "message_glove"
#     model_file = project_directory_path + "/models/Transformer_model_" + name
#     # 从指定路径加载模型
#     model = load_model(model_file)
#
#     y_pred = model.predict(data)
#     y_pred = np.round(np.squeeze(y_pred)).astype(int)
#
#     log_message(f"[Predict result] : {y_pred} " + "This is a spam email." if y_pred else "This is not a spam email.")
#     return True if y_pred else False