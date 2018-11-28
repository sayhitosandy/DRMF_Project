from data_manager import Data_Factory
from rating_models import PMF
import pickle
import sys
import os

data_factory = Data_Factory()

data = sys.argv[1]
file = data.split('/')[2].split('.')[0]
test_file = "./test/" + file + "/"
output_file = "./outputs/" + file + "/" 
if not os.path.exists(output_file):
    os.makedirs(output_file)

binary_rating = False

R = pickle.load(open(test_file + 'ratings.all', 'rb'))

train_user = data_factory.read_rating(test_file + 'train_user.dat', binary_rating)
train_item = data_factory.read_rating(test_file + 'train_item.dat', binary_rating)
valid_user = data_factory.read_rating(test_file + 'valid_user.dat', binary_rating)
test_user = data_factory.read_rating(test_file + 'test_user.dat', binary_rating)

'''PMF'''
PMF(res_dir=output_file, lambda_u=0.1, lambda_v=0.1, train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)