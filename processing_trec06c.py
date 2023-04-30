"""
处理trec06c数据集
"""
import csv
import email
import jieba
import re

from multiprocessing import Pool

# 定义表头
headers = ["label", "from", "to", "subject", "words"]

# 定义一个字典映射，将spam和ham分别对应为1和0
label_map = {"spam": 1, "ham": 0}


def process_line(line):
    # 获取标签和路径
    label, path = line.strip().split()
    # 将标签转换为0或1，使用字典映射（也可以使用if-else语句）
    label = label_map[label]
    path = "trec06c" + path[2:]
    # 读取邮件文件
    with open(path, "r", encoding="gb18030", errors="ignore") as f3:
        msg = email.message_from_file(f3)
        # 获取发件人、收件人、主题和正文
        from_ = msg.get("From")
        to = msg.get("To")
        subject = msg.get("Subject")
        payload = msg.get_payload()
        # 删除删除不可读字符
        chinese_only = re.sub(r'[^\u4e00-\u9fa5]', '', payload)

        # 从文件中读取停用词列表并转换为集合（优化点1）
        with open('baidu_stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f)

        # 对正文进行分词并过滤掉停用词（优化点2）
        filtered_words = [word for word in jieba.lcut(chinese_only) if word not in stopwords]

        if filtered_words:  # 添加判断语句，当filtered_words不为空时才写入数据。
            return {"label": label, "from": from_, "to": to, "subject": subject, "words": filtered_words}


if __name__ == '__main__':
    with open("trec06c/full/index", "r") as f1, open("trec06c.csv", "w", newline="", encoding="utf-8") as f2:
        # 创建一个字典写入器对象
        writer = csv.DictWriter(f2, fieldnames=headers)
        # 写入表头
        writer.writeheader()

        pool = Pool()  # 创建进程池

        results = pool.map(process_line, f1)  # 使用多进程处理数据

        for result in results:
            if result:  # 判断结果是否为空，如果不为空则写入数据。
                writer.writerow(result)
