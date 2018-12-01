import numpy as np
import json
from data_manager import Data_Factory
import os
import numpy as np
import pickle
import sys
from scipy.sparse import csr_matrix

# MAKE ITEM.DAT
data_factory = Data_Factory()
with open('FlickscoreData-26oct2018/movies.json') as f:
    movies=f.readlines()

movie_dic={}
movie_counter=0
movie_ids=[]
movie_des=[]

for line in movies:
    movie_data=json.loads(line)
    movie_dic[movie_data['movie_id']]=movie_counter
    movie_des.append(movie_data['description'])
    movie_ids.append(movie_counter)
    movie_counter+=1

filestr=""
for i in range(len(movie_ids)):
    writeline=str(movie_ids[i])+"::"+str(movie_des[i])+"\n"
    filestr+=writeline
with open('items.dat', 'a') as f:
    f.write(filestr)

# MAKE USER.DAT

with open('FlickscoreData-26oct2018/users.json') as f:
    users=f.readlines()

user_dic={}
user_counter=0
user_des=[]

for line in users:
    user_data=json.loads(line)
    user_dic[user_data['_id']]=user_counter
    try:
    	user_des.append(str(user_data['languages'][0]) + ' ' + str(user_data['job']) + ' ' + str(user_data['state']))
    except:
    	user_des.append("")
    user_counter+=1

user_ids = list(range(user_counter))
filestr=""
for i in user_ids:
    writeline=str(user_ids[i])+"::"+str(user_des[i])+"\n"
    filestr+=writeline
with open('users.dat', 'w') as f:
    f.write(filestr)
f.close()
# MAKE RATINGS.DAT

with open('FlickscoreData-26oct2018/ratings.json') as f:
    e=f.readlines()

user_ids=[]
movie_ids=[]
ratings=[]
user_count = 0

for line in e:
    movie_data=json.loads(line)
    userid=movie_data['_id']
    userid=user_dic[userid]
    rated=movie_data['rated']
    for k,v in rated.items():
        if (v[0]!='submit' and v[0]!='submitexit' and v[0]!='submitmore' and v[0]!=''):
            user_ids.append(userid)
            k=movie_dic[k]
            movie_ids.append(k)
            ratings.append(str(float(v[0])+2))
mapped_ids = sorted(list(set(movie_ids)))
for i in range(len(movie_ids)):
    movie_ids[i] = mapped_ids.index(movie_ids[i])
mapped_ids = sorted(list(set(user_ids)))
for i in range(len(user_ids)):
    user_ids[i] = mapped_ids.index(user_ids[i])
filestr=""
for i in range(len(user_ids)):
    writeline=str(user_ids[i])+"::"+str(movie_ids[i])+"::"+str(ratings[i])+"\n"
    filestr+=writeline

with open('ratings.dat', 'w') as f:
    f.write(filestr)

data=np.array([float(i) for i in ratings])
row=np.array([int(i) for i in user_ids])
col=np.array([int(i) for i in movie_ids])
R = csr_matrix((data, (row, col)))
f = open('test/FS/ratings.all', 'wb')
pickle.dump(R, f)
f.close()

f = open("test/FS/ratings.all", 'rb')
R = pickle.load(f)
f.close()

split_ratio = 0.2


##R = csr_matrix(([int(i) for i in ratings], ([int(i) for i in user_ids], [int(i) for i in movie_ids])))
data_factory.generate_train_valid_test_file_from_R('test/FS/', R, split_ratio)
