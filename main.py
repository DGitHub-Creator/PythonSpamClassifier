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

import io

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.models import load_model

# 1. 加载模型
model_name = "LSTM"
model_file = project_directory_path + f"/models/{model_name}_model_message_glove"
model = load_model(model_file)

# 2. 创建日志文件写入器
log_dir = './logs/'
file_writer = tf.summary.create_file_writer(log_dir)

# 3. 将模型结构写入日志文件
with file_writer.as_default():
    # 将模型结构转换为图片
    img_data = tf.keras.utils.plot_model(model, show_shapes=True, rankdir='TB', expand_nested=True, dpi=96)

    # 将图片数据转换为PIL.Image对象
    img = Image.open(io.BytesIO(img_data))

    # 将PIL.Image对象转换为NumPy数组
    img_array = np.array(img).astype(np.uint8)

    # 将数组添加到日志文件中
    img_tensor = tf.expand_dims(img_array, 0)
    tf.summary.image("Model Architecture", img_tensor, step=0)
