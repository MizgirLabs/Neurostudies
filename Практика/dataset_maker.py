# Всего различных иероглифов: 250
# Из них в тренировочной выборке: 192
# В тестовой: 58

import sqlite3
import random

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

def data_base():
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
    print(rand_sample)
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

    conn.close()
