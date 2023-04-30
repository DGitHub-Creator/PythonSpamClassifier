"""
处理trec06p数据集
"""
import csv
import email
import re

from multiprocessing import Pool

# 定义表头
headers = ["label", "from", "to", "subject", "words"]

# 定义一个字典映射，将spam和ham分别对应为1和0
label_map = {"spam": 1, "ham": 0}


def extract_text(text):
    pattern = r'<(.+?)>'  # 匹配 '<' 和 '>' 中间的任何字符
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return text


def get_first_text_block(email_message):
    if email_message.is_multipart():
        return get_first_text_block(email_message.get_payload(0))
    else:
        return email_message.get_payload(None, True)


def process_line(line):
    # 获取标签和路径
    label, path = line.strip().split()
    # 将标签转换为0或1，使用字典映射（也可以使用if-else语句）
    label = label_map[label]
    path = "trec06p" + path[2:]
    # 读取邮件文件
    with open(path, "r", encoding="utf-8", errors="ignore") as f3:
        try:
            msg = email.message_from_file(f3)
            # 获取发件人、收件人、主题和正文
            from_ = msg.get("From")
            filtered_from_ = extract_text(from_)
            to = msg.get("To")
            filtered_to = extract_text(to)
            subject = msg.get("Subject")
            payload = get_first_text_block(msg)
            # 将payload存入csv文件中
            with open('output.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([payload])
            # 删除不可读字符
            # filtered_words = re.sub(r'[^\u4e00-\u9fa5]', '', payload)
            # 添加打印语句，显示当前正在处理的文件名
            print("Processing file:", path)
            return {"label": label, "from": filtered_from_, "to": filtered_to, "subject": subject, "words": payload}
        except:
            return None


if __name__ == '__main__':
    with open("trec06p/full/index", "r") as f1, open("trec06p.csv", "w", newline="", encoding="utf-8") as f2:
        # 创建一个字典写入器对象
        writer = csv.DictWriter(f2, fieldnames=headers)
        # 写入表头
        writer.writeheader()

        pool = Pool()  # 创建进程池

        for result in pool.imap_unordered(process_line, f1):
            if result:  # 判断结果是否为空，如果不为空则写入数据。
                writer.writerow(result)
