import csv
import os
import numpy as np
import pickle
from scipy.sparse.csr import csr_matrix
import sys

data = sys.argv[1]
file = data.split('/')[2].split('.')[0]

# Read ratings matrix
with open(data, 'r') as f:
	r = np.loadtxt(f, skiprows=1, dtype='str')
r = np.array(np.delete(r, 0, 1), dtype=np.int32)

# Read drug (item) names
with open(data, 'r') as f:
	users = np.loadtxt(f, skiprows=1, usecols=(0,), dtype='str')

# Read target (user) names
f = open(data, 'r')
line = f.readline()
items = []
for i in line.split('\t'):
	if (i != ""):
		items.append(i.strip())
f.close()

items = np.array(items, dtype='str')

test_file = "./test/" + file + "/" 
if not os.path.exists(test_file):
    os.makedirs(test_file)

np.save(test_file + 'user_ids.npy', users)
np.save(test_file + 'item_ids.npy', items)

f = open(test_file + 'ratings.dat', 'w')
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