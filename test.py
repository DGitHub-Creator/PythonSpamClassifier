import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

fg = "full"
email_data = pd.read_csv(f"cleaned_email_{fg}.csv", usecols=["body", "spam"])

# 定义训练数据
data = email_data['body']
labels = email_data['spam']

half_size = len(data) // 10

data = data.iloc[:half_size]
labels = labels.iloc[:half_size]

# # 定义测试数据
# test_data = [
#     "This is a positive sentence.",
#     "This is another positive sentence.",
#     "This is a negative sentence.",
#     "This is another negative sentence."
# ]

# 设置管道
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])

# 设置网格搜索的参数
parameters = {
    'vect__max_df': [0.5, 0.75, 1.0],
    'vect__min_df': [1, 2, 5],
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'vect__token_pattern': [r'\b\w\w+\b', r'\b\w{2,}\b'],
    'clf__alpha': [0.01, 0.1, 1.0],
}

# 使用网格搜索法搜索最优参数
grid_search = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1, verbose=1)
grid_search.fit(data, labels)

# 输出最优参数
print("Best parameters set:")
print(grid_search.best_params_)
