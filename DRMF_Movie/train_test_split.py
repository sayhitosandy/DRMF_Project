from data_manager import Data_Factory
import os
import numpy as np
import pickle
import sys

data_factory = Data_Factory()
'''
data = sys.argv[1]
file = data.split('/')[2].split('.')[0]
test_file = "./test/" + file + "/" 
if not os.path.exists(test_file):
    os.makedirs(test_file)
'''

f = open('test/FS/ratings.dat', 'w')
user = []
item = []
rating = []
for u in range(r.shape[0]):
	for i in range(r.shape[1]):
		user.append(u)
		item.append(i)
		rating.append(r[u][i])
		f.write(str(u) + "::" + str(i) + "::" + str(float(r[u][i])) + "\n")
f.close()

R = csr_matrix((rating, (user, item)))
f = open(test_file + 'ratings.all', 'wb')
pickle.dump(R, f)
f.close()



f = open("test/FS/ratings.dat", 'r')
R = pickle.load(f)
f.close()

split_ratio = 0.2

data_factory.generate_train_valid_test_file_from_R(test_file, R, split_ratio)