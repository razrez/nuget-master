#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import sys
# import csv
# import requests
# import json
# import time

# # %pip install re
# # %pip install nltk
# # %pip install unicodedata
# # %pip install contractions
# # %pip install inflect
# # %pip install emoji

# import re
# import os
# import nltk
# import emoji
# import unicodedata
# import contractions
# import inflect
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

# # для этого тож pip install нужно прописать
# # nltk.download('stopwords')
# # nltk.download('punkt')
# # nltk.download('averaged_perceptron_tagger')

# # Подготовка к частеречной разметке текста путём установки библиотеки Spacy, загрузки perceptron_tagger и модуля Spacy en
# # %pip install spacy
# #!python -m spacy download en_core_web_sm
# import spacy

# # Commented out IPython magic to ensure Python compatibility.
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import sys
# import warnings
# warnings.filterwarnings('ignore')
# import re
# import csv
# from sklearn.cluster import KMeans

# import sys
# np.set_printoptions(suppress=True)
# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(precision=3)

# import warnings
# warnings.filterwarnings('ignore')
# import joblib

# # %pip install nltk
# # %pip install unicodedata
# # %pip install contractions
# # %pip install inflect
# # %pip install emoji


# # from sklearn.metrics.pairwise import cosine_similarity
# # import gensim
# # pip install scipy==1.12 !!!!!!!!!!!!!
# # pip install scikit-learn==1.2.2 !!!!!
# from gensim.models import FastText
# from sklearn.neighbors import NearestNeighbors

# """Препроцессинг"""

# # Функция для очистки текста
# def clean_text(input_text):

#     # HTML-теги: первый шаг - удалить из входного текста все HTML-теги
#     clean_text = re.sub('<[^<]+?>', '', input_text)

#     # URL и ссылки: далее - удаляем из текста все URL и ссылки
#     clean_text = re.sub(r'http\S+', '', clean_text)

#     # Эмоджи и эмотиконы: удаляем их нафиг, это кринж какой-то бессмысленный, только шума добавят
#     clean_text = remove_emojis(clean_text)

#     # Приводим все входные данные к нижнему регистру
#     clean_text = clean_text.lower()

#     # Убираем все пробелы
#     # Так как все данные теперь представлены словами - удалим пробелы
#     clean_text = re.sub(r'\s+', ' ', clean_text)

#     # Преобразование символов с диакритическими знаками к ASCII-символам: используем функцию normalize из модуля unicodedata и преобразуем символы с диакритическими знаками к ASCII-символам
#     clean_text = unicodedata.normalize('NFKD', clean_text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

#     # Разворачиваем сокращения: текст часто содержит конструкции вроде "don't" или "won't", поэтому развернём подобные сокращения
#     clean_text = contractions.fix(clean_text)

#     # Специальные случаи для языков программирования
#     clean_text = re.sub(r'c\#', 'csharp', clean_text)
#     clean_text = re.sub(r'c\+\+', 'cpp', clean_text)

#     # Убираем специальные символы: избавляемся от всего, что не является "словами"
#     clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', clean_text)

#     # Записываем числа прописью: 100 превращается в "сто" (для компьютера)
#     temp = inflect.engine()
#     words = []
#     for word in clean_text.split():
#         if word.isdigit():
#             words.append(temp.number_to_words(word))
#         else:
#             words.append(word)
#     clean_text = ' '.join(words)

#     # Стоп-слова: удаление стоп-слов - это стандартная практика очистки текстов
#     stop_words = set(stopwords.words('english'))
#     tokens = word_tokenize(clean_text)
#     tokens = [token for token in tokens if token not in stop_words]
#     clean_text = ' '.join(tokens)

#     # Add full-stop to end of sentences
#     clean_text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', clean_text)

#     # Знаки препинания: далее - удаляем из текста все знаки препинания
#     clean_text = re.sub(r'[^\w\s]', '', clean_text)

#     # И наконец - возвращаем очищенный текст
#     return clean_text


# # Функция для преобразования эмоджи в слова
# def emojis_to_txt(text):

#     # Модуль emoji: преобразование эмоджи в их словесные описания
#     clean_text = emoji.demojize(text, delimiters=(" ", " "))

#     # Редактирование текста путём замены ":" и" _", а так же - путём добавления пробела между отдельными словами
#     clean_text = clean_text.replace(":", "").replace("_", " ")

#     return clean_text


# def remove_emojis(text):
#     emoji_pattern = re.compile(
#         "["
#         "\U00010000-\U0010FFFF"  # Unicode characters beyond the Basic Multilingual Plane
#         "\u200d"  # Zero-width joiner
#         "\u2640-\u2642"  # Male and female signs
#         "\u2600-\u26FF"  # Miscellaneous symbols
#         "\u2700-\u27BF"  # Dingbats
#         "]+",
#         flags=re.UNICODE
#     )
#     return emoji_pattern.sub(r'', text)


# # %pip install spacy
# # nltk.download('averaged_perceptron_tagger')
# #!python -m spacy download en_core_web_sm

# nlp = spacy.load('en_core_web_sm')
# lemmatize_exceptions = ['ios', 'mmorts']

# def lemmatize_and_postag(input_text):

#     doc = nlp(input_text)
#     lemmatize_output = []

#     # Iterate over each token in the document
#     for token in doc:
#         if token.text in lemmatize_exceptions:
#             lemmatize_output.append(token.text)
#             # lemmatize_output.append(token.text + '_' + token.pos_)
#         else:
#             # Append the token lemma and its POS tag to the tagged_output list
#             lemmatize_output.append(token.lemma_)

#     # Join the tagged_output list into a single string
#     lemmatize_output_str = ' '.join(lemmatize_output)

#     return lemmatize_output_str


# def preprocesse(input_text):
#     clear_text = clean_text(input_text)

#     tagged_output = lemmatize_and_postag(clear_text)

#     return tagged_output


# # Preprocess the text data
# def preprocess_query(query_text):
#     query_text = preprocesse(query_text)
#     return query_text.split()

# # get the vector representation of a query
# def get_query_vector(query, model):
#     words = preprocess_query(query)
#     query_vector = model.wv.get_sentence_vector(words)
#     return query_vector

# # get the vector representation of a repository
# def get_repository_vector(description, model):
#     words = description
#     repo_vector = model.wv.get_sentence_vector(words)
#     return repo_vector

# # function to retrieve relevant repositories for a given query
# def retrieve_relevant_repositories(query, knn, model, data):
#     query_vector = get_query_vector(query, model)
#     distances, indices = knn.kneighbors([query_vector])
#     return data.Repository.iloc[indices[0]].values.tolist()


# current_dir = os.path.dirname(__file__)

# """FastText + kNN(kd-tree)"""
# """Загрузка моделей и их использование"""
# # data = pd.read_csv(r'C:\Users\sharp\Desktop\nuget-master\src\nuget-master\src\python-scripts\dataset\preprocessed_txtdata.csv')
# data = pd.read_csv(os.path.join(current_dir, 'dataset', 'preprocessed_txtdata.csv'))
# data['Description'] = data['Description'].apply(lambda x: [word.split('_')[0] for word in x.split()])

# # fasttext_model = FastText.load(r"C:\Users\sharp\Desktop\nuget-master\src\nuget-master\src\python-scripts\model\trained_models\fasttext_model")
# # knn = joblib.load(r'C:\Users\sharp\Desktop\nuget-master\src\nuget-master\src\python-scripts\model\trained_models\knn_model.joblib')
# fasttext_model = FastText.load(os.path.join(current_dir, 'model', 'trained_models', 'fasttext_model'))
# knn = joblib.load(os.path.join(current_dir, 'model', 'trained_models', 'knn_model.joblib'))

# while True:
#     try:
#         line = sys.stdin.readline().strip()
#         ##line = sys.argv[1]
#         if line:
#             query = line
#             relevant_repositories = retrieve_relevant_repositories(query, knn, fasttext_model, data)
#             print(",".join(relevant_repositories))

#     except Exception as e: print("error")

###### new version
## pip install sentence-transformers scikit-learn pandas numpy
import os
import sys
import re
import unicodedata
import joblib
import spacy
import emoji
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

###############################################################################
# 0. Минимальные функции очистки текста
###############################################################################
def emojis_to_txt(text):
    """
    Модуль emoji: преобразование эмоджи в их словесные описания
    Например, 😊 -> :smiling_face_with_smiling_eyes:
    """
    clean_text = emoji.demojize(text, delimiters=(" ", " "))
    # Убираем двоеточия и подчёркивания, чтобы текст был более читаем
    clean_text = clean_text.replace(":", "").replace("_", " ")
    return clean_text

def remove_emojis(text):
    """
    Удаление символов Юникода, относящихся к эмоджи, символам и т.д.
    """
    emoji_pattern = re.compile(
        "["
        "\U00010000-\U0010FFFF"  # Unicode beyond BMP
        "\u200d"                 # Zero-width joiner
        "\u2640-\u2642"          # Male/female signs
        "\u2600-\u26FF"          # Misc. symbols
        "\u2700-\u27BF"          # Dingbats
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

###############################################################################
# 1. Инициализация spaCy для лемматизации (при желании)
###############################################################################
nlp = spacy.load('en_core_web_sm')
# если нужно исключить «ненужные» слова, добавьте их в exceptions
lemmatize_exceptions = ['ios', 'mmorts']

def lemmatize_text(input_text):
    """
    Лемматизируем (приводим слова к словарной форме)
    """
    doc = nlp(input_text)
    output_tokens = []
    for token in doc:
        if token.text in lemmatize_exceptions:
            output_tokens.append(token.text)
        else:
            output_tokens.append(token.lemma_)
    return " ".join(output_tokens)

###############################################################################
# 2. Функция «общей» очистки текста
###############################################################################
def clean_text(input_text):
    # Сначала превращаем эмоджи в «:smile:»
    demojized = emojis_to_txt(input_text)
    # Потом удаляем «настоящие» эмоджи, которые могут остаться
    no_emoji = remove_emojis(demojized)
    # Normalize unicode (NFKD), убрать диакритику, если хотите
    # no_emoji = unicodedata.normalize('NFKD', no_emoji).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Лемматизация
    lemmatized = lemmatize_text(no_emoji)
    return lemmatized.lower().strip()

###############################################################################
# 3. Предобработка запроса (с помощью clean_text)
###############################################################################
def preprocess_query(query_text):
    """
    Выполняем очистку (эмоджи, лемма...) + 
    здесь можно токенизировать, но Sentence-BERT сам умеет разбивать текст
    """
    return clean_text(query_text)

###############################################################################
# 4. Загрузка модели и kNN индекса
###############################################################################
# Предположим, что скрипт лежит рядом с папкой model/trained_models
current_dir = os.path.dirname(__file__)

# - Модель Sentence-BERT (ранее сохраненная)
#   Вы должны иметь папку sbert_model_folder, где лежат config.json, etc.
model_path = os.path.join(current_dir, 'model', 'trained_models', 'sbert_model_folder')
model = SentenceTransformer(model_path)

# - kNN индекс (NearestNeighbors), сохранённый joblib-ом
knn_path = os.path.join(current_dir, 'model', 'trained_models', 'knn_model.joblib')
knn = joblib.load(knn_path)

# 5. Загрузка базы репозиториев (CSV)
data_path = os.path.join(current_dir, 'dataset', 'preprocessed_txtdata.csv')
data = pd.read_csv(data_path)



# Предположим, что в data есть колонки:
#   Repository  (название)
#   Description (текст, либо токенизированный)
# Если Description у вас массив токенов, а S-BERT принимает строку, 
# можете склеить их: " ".join(Description)

###############################################################################
# 6. Функция поиска ближайших репозиториев
###############################################################################
def retrieve_relevant_repositories(query, knn, model, data):
    # Предобрабатываем
    clean_q = preprocess_query(query)
    # Получаем эмбеддинг через модель
    query_vector = model.encode([clean_q])  # shape: (1, dim)

    # Ищем ближайших
    distances, indices = knn.kneighbors(query_vector, n_neighbors=5)
    # Возвращаем список названий (или full rows)
    repos = data.Repository.iloc[indices[0]].values.tolist()
    return repos

###############################################################################
# 7. Основной цикл чтения query из stdin, выводим top-5
###############################################################################
if __name__ == "__main__":

    while True:
        try:
            line = sys.stdin.readline().strip()
            if not line:
                break  # Пустая строка -> выходим

            query = line
            relevant = retrieve_relevant_repositories(query, knn, model, data)
            # Выводим списком через запятую
            print(",".join(relevant))

        except Exception as e:
            print(f"error: {e}")
            break
