# Всего различных иероглифов: 250
# Из них в тренировочной выборке: 192
# В тестовой: 58
# таблицы в базе данных: characters, query, train

import sqlite3
import random
import numpy as np

def dict_maker():
    d = {}  # ключ - иероглиф, значение - количество повторений

    with open('query.txt', 'r', encoding='utf-8') as query:
        query_data = query.read()
    query_arr = query_data.split('\n')
    for phrase in query_arr:
        for char in phrase:
            if char not in d:
                d[char] = 1
            else:
                d[char] += 1

    with open('train.txt', 'r', encoding='utf-8') as train:
        train_data = train.read()
    train_arr = train_data.split('\n')
    for phrase in train_arr:
        for char in phrase:
            if char not in d:
                d[char] = 1
            else:
                d[char] += 1
    return d

def characters_base():
    d = dict_maker()
    conn = sqlite3.connect('characters.db')
    c = conn.cursor()
    c.executescript("""DROP TABLE IF EXISTS characters;
    
             CREATE TABLE characters
             (character TEXT, 
             frequency INTEGER,
             sample FLOAT, 
             normalized_sample FLOAT);
                   """)

# frequency - сколько раз встречается в выборках
# sample - избавляемся от повторяющихся значений, не слишком влияя на частоту
# normalized_sample - ставим числа в диапазон от 0.01 до 1.0 для формирования входных сигналов

    rand_sample = [random.uniform(0, 1) for i in range(len(d))]
    for i, key in enumerate(d):
        c.execute('''
        INSERT INTO characters (character, frequency, sample, normalized_sample) 
        VALUES (?, ?, ?, ?)
            ''', [key, d[key], d[key] + rand_sample[i], ((d[key] + rand_sample[i]) / 55 * 0.99) + 0.01])
    c.execute('''SELECT character, frequency, sample, normalized_sample
                FROM characters 
                ORDER BY frequency
                ''')

    result = c.fetchall()
    conn.commit()
    conn.close()


# векторизация тестовой выборки
# cоздание новой таблицы - фраза|векторизованный вариант
# векторизованная строка - ТЕКСТ!

def query_base():
    conn = sqlite3.connect('characters.db')
    c = conn.cursor()
    with open('query.txt', 'r', encoding='UTF-8') as q:
        query = q.read().split()
    q_vectorized = []
    for row in query:
        line = []
        for char in row:
            c.execute('''SELECT normalized_sample
                    FROM characters 
                    WHERE character == ?
                    ''', [char])
            result = c.fetchall()
            line.append(result[0][0])
        q_vectorized.append(line)

    c.executescript("""DROP TABLE IF EXISTS query;

                 CREATE TABLE query
                 (phrase TEXT, 
                 vectorization,
                 target TEXT, 
                 arr_target);
                       """)
    for i in range(len(query)):
        c.execute('''INSERT INTO query (phrase, vectorization)
                    VALUES (?, ?)''',
                  [query[i], ' '.join(map(str, q_vectorized[i]))])
    conn.commit()
    conn.close()

# тренировочная база
# фраза|векторизация иероглифов в фразе|фраза с разбивкой на слова|целевой выход
# target придется делать вручную

def train_base():
    conn = sqlite3.connect('characters.db')
    c = conn.cursor()
    with open('train.txt', 'r', encoding='UTF-8') as t:
        train = t.read().split()
    train[0] = train[0].replace('\ufeff', '')
    t_vectorized = []
    for row in train:
        line = []
        for char in row:
            c.execute('''SELECT normalized_sample
                        FROM characters 
                        WHERE character == ?
                        ''', [char])
            result = c.fetchall()
            line.append(result[0][0])
        t_vectorized.append(line)

    c.executescript("""DROP TABLE IF EXISTS train;

                 CREATE TABLE train
                 (phrase TEXT,
                 vectorization,
                 target TEXT, 
                 arr_target);
                       """)
    for i in range(len(train)):
        c.execute('''INSERT INTO train (phrase, vectorization, target)
                    VALUES (?, ?, ?)''',
                  [train[i], ' '.join(map(str, t_vectorized[i])), train[i]])
    conn.commit()
    conn.close()


def vec_train(): # вставляю в базу данных векторизованный вариант обучающей выборки
    conn = sqlite3.connect('characters.db')
    c = conn.cursor()
    c.execute('''SELECT target
                 FROM train
                    ''')
    result = c.fetchall()
    for line in result:
        vector = []
        for el in line[0]:
            if el == ' ':
                vector.append('0.01')
            else:
                vector.append('0.99')
        c.execute('''UPDATE train 
                     SET arr_target = (?)
                     WHERE target = (?)''',
                  [' '.join(vector), line[0]])
    conn.commit()
    conn.close()

def vec_query(): # вставляю в базу данных векторизованный вариант тестовой выборки
    conn = sqlite3.connect('characters.db')
    c = conn.cursor()
    c.execute('''SELECT target
                 FROM query
                        ''')
    result = c.fetchall()
    for line in result:
        vector = []
        for el in line[0]:
            if el == ' ':
                vector.append('0.01')
            else:
                vector.append('0.99')
        c.execute('''UPDATE query 
                     SET arr_target = (?)
                     WHERE target = (?)''',
                  [' '.join(vector), line[0]])
    conn.commit()
    conn.close()


# ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ И ТЕСТА

# массив кортежей ([дано], [цель]), тренировочные данные
def train_set():
    conn = sqlite3.connect('characters.db')
    c = conn.cursor()
    c.execute('''SELECT vectorization, arr_target
                 FROM train
                            ''')
    result = c.fetchall()
    train_arr = [] # словарь - чтобы порядок фраз не был фиксированным
    for line in result:
        train_arr.append((
            np.array([float(x) for x in line[0].split()]),
            np.array([float(x) for x in line[1].split()])))
    return train_arr


# массив кортежей ([дано], [цель])
def query_set():
    conn = sqlite3.connect('characters.db')
    c = conn.cursor()
    c.execute('''SELECT vectorization, arr_target
                 FROM query
                                ''')
    result = c.fetchall()
    query_arr = []  # словарь - чтобы порядок фраз не был фиксированным
    for line in result:
        query_arr.append((
            np.array([float(x) for x in line[0].split()]),
            np.array([float(x) for x in line[1].split()])))
    return query_arr