from data_manager import Data_Factory
import os
import numpy as np
import pickle
import sys

data_factory = Data_Factory()

data = sys.argv[1]
file = data.split('/')[2].split('.')[0]
test_file = "./test/" + file + "/" 
if not os.path.exists(test_file):
    os.makedirs(test_file)

f = open(test_file + 'ratings.all', 'rb')
R = pickle.load(f)
f.close()

split_ratio = 0.2

data_factory.generate_train_valid_test_file_from_R(test_file, R, split_ratio)