import hashlib
import os

import pandas as pd


def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calculate_md5_for_all_files_in_directory(directory):
    md5_values = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            md5_value = calculate_md5(file_path)
            md5_values[file_path] = md5_value
    return md5_values


directory = './data'
md5_values = calculate_md5_for_all_files_in_directory(directory)
for file_path, md5_value in md5_values.items():
    print(f"{file_path}: {md5_value}")
#
# email_data = pd.read_csv("data/email_06_full.csv")
# print(email_data["body"][0])
