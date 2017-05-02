import sqlite3
import gensim

conn = sqlite3.connect('movies.sqlite')
c = conn.cursor()

# SQL 쿼리 실행
c.execute("select * from comments")

# 데이타 Fetch
rows = c.fetchall()

sentences = []

idx = 0
for a_row in rows:
    words = a_row[3].split()
    sentences.append(words)

model = gensim.models.Word2Vec(sentences, window=5, min_count=5, size=100)

print(model.syn0.shape)