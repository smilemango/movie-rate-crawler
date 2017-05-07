import sqlite3
import gensim


conn = sqlite3.connect('movies.sqlite')
c = conn.cursor()

# SQL 쿼리 실행
c.execute("select * from comments order by comment_id limit 10 ")

# 데이타 Fetch
rows = c.fetchall()

sentences = []

idx = 0
for a_row in rows:
    words = a_row[3].split()
    sentences.append(words)

print("SENTENSE SIZE : %d " %  len(sentences) )

model = gensim.models.Word2Vec(sentences, window=10, min_count=1, size=3)

print (model.most_similar('하정우'))

X = model[model.wv.vocab]

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import font_manager, rc
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pylab

# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
# rc('font', family=font_name)
krfont = {'family' : 'nanumgothic', 'weight' : 'bold', 'size' : 10}
rc('font', **krfont)


tsne = TSNE(n_components=3)
X_tsne = tsne.fit_transform(X)

fig = pylab.figure()
ax = Axes3D(fig)

ax.set_title('암살 영화에 대한 단어 임베딩')
#ax.set_xlabel('xlabel')
#ax.set_ylabel('ylabel')
#ax.set_zlabel('zlabel')




#ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2])


for idx, x_val in enumerate(X):
    ax.scatter(x_val[0], x_val[1], x_val[2])
    ax.text(x_val[0],x_val[1],x_val[2], model.wv.index2word[idx])
    if model.wv.index2word[idx] == '하정우':
        ax.text(x_val[0],x_val[1],x_val[2], "하정우(%f,%f,%f)" % ( x_val[0], x_val[1], x_val[2]), color='red')
    if model.wv.index2word[idx] == '전지현':
        ax.text(x_val[0],x_val[1],x_val[2], "전지현(%f,%f,%f)" % ( x_val[0], x_val[1], x_val[2]), color='red')
    if model.wv.index2word[idx] == '이정재':
        ax.text(x_val[0],x_val[1],x_val[2], "이정재(%f,%f,%f)" % ( x_val[0], x_val[1], x_val[2]), color='red')

pyplot.show()




