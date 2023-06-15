# import platform
# import re
# import subprocess
# import sys
#
# import keras
# import numpy as np
# import pandas as pd
# import sklearn
# import tensorflow as tf
#
# # 获取操作系统信息
# operating_system = platform.platform()
#
# # 获取Python版本信息
# python_version = sys.version
#
# # 获取TensorFlow版本信息
# tensorflow_version = tf.__version__
#
# # 获取Keras版本信息
# keras_version = keras.__version__
#
# # 获取Scikit-learn版本信息
# sklearn_version = sklearn.__version__
#
# # 获取Pandas版本信息
# pandas_version = pd.__version__
#
# # 获取NumPy版本信息
# numpy_version = np.__version__
#
# # 获取GPU设备名称
# gpu_device_name = tf.test.gpu_device_name()
#
# # 获取CUDA版本信息
# cuda_version = None
# try:
#     result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
#     output = result.stdout.strip()
#
#     # 使用正则表达式提取CUDA版本信息
#     cuda_version_match = re.search(r'release (\d+\.\d+)', output)
#     cuda_version = cuda_version_match.group(1) if cuda_version_match else None
# except FileNotFoundError:
#     pass
#
# # 打印版本信息
# print("操作系统：", operating_system)
# print("Python版本：", python_version)
# print("CUDA版本：", cuda_version)
# print("TensorFlow版本：", tensorflow_version)
# print("Keras版本：", keras_version)
# print("Scikit-learn版本：", sklearn_version)
# print("Pandas版本：", pandas_version)
# print("NumPy版本：", numpy_version)


# 数据集	邮件总数	垃圾邮件数	正常邮件数	语言
# trec06p	37,822	25,220	12,602	英文
# trec07p	75,419	50,199	25,220	英文


from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np

# 输入的文本
text = "Do you have the knowledge and the experience but lack the qualifications?"

# 分词
tokens = word_tokenize(text)

# 计算每个单词的频率
word_counts = Counter(tokens)

# 创建一个字典，将每个单词映射到一个唯一的索引
word2index = {word: i for i, word in enumerate(word_counts.keys())}

# 创建一个向量表示文本
vector = np.zeros(len(word2index))

for word, count in word_counts.items():
    # 使用单词在字典中的索引来更新向量
    vector[word2index[word]] = count

print("Word2Index:", word2index)
print("Vector Representation:", vector)
