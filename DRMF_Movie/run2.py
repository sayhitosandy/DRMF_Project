'''
Created on Dec 9, 2015
@author: donghyun

Modified on June 20, 2017 
@author: Hao Wu, Zhengxin Zhang

'''
import argparse
import os
import pickle
import sys
from data_manager import Data_Factory
from util import index_to_input, histogram
from rating_models import DRMF

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

max_iter=200
vocab_size=200

init_W=None
emb_dim=200
maxlen_doc_x=200
maxlen_sent_x=200


DRMF(max_iter=max_iter, res_dir=output_file, lambda_u=10, lambda_v=1, dimension=50, vocab_size=vocab_size, init_W=init_W, give_weight=1, DNN_X=None, DNN_Y=None, emb_dim=emb_dim, num_kernel_per_ws=100, dropout_rate=0.2, train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R, dnn_type='CNN', reg_schema='Item', maxlen_doc=[maxlen_doc_x, 0], maxlen_sent=[maxlen_sent_x, 0])
exit()

if(1==2):    
    '''PMF'''
    PMF(max_iter=max_iter, res_dir=res_dir, lambda_u=0.1, lambda_v=0.1, dimension=dimension,
                  train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R)    
          
    # DNN-Regularized Matrix Factorization
    vocab_size = len(D_all['XY_vocab']) + 1

    if pretrain_w2v is None:
        init_W = None
    else:
        init_W = data_factory.read_pretrained_word2vec(pretrain_w2v, D_all['XY_vocab'], emb_dim)
   
    '''for items'''
    DNN_X = D_all['X_sequence2D']
    len_doc_x = [len(profile) for profile in DNN_X]
    len_sent_x = []
    for profile in DNN_X:
        for review in profile:
            len_sent_x.append(len(review))
     
    # histogram('ITEM: Number of sequences',len_doc_x)         
    # histogram('ITEM: Length of sequences',len_sent_x)
    len_doc_x = sorted(len_doc_x)
    len_sent_x = sorted(len_sent_x)
    maxlen_doc_x = len_doc_x[(int)(len(len_doc_x) * threshold_doclen)]
    maxlen_sent_x = len_sent_x[(int)(len(len_sent_x) * threshold_sentlen)]
    print ("X_sequence2D, maxlen_doc:%d maxlen_sent:%d" % (maxlen_doc_x, maxlen_sent_x))
    DNN_X = index_to_input(DNN_X, maxlen_doc_x, maxlen_sent_x)
    
    
    '''DRMF-CNN-Item (namely, ConvMF)'''
    DRMF(max_iter=max_iter, res_dir=res_dir, lambda_u=10, lambda_v=1, dimension=50, vocab_size=vocab_size, init_W=init_W, give_weight=give_weight, DNN_X=DNN_X, DNN_Y=None,
                     emb_dim=emb_dim, num_kernel_per_ws=100, dropout_rate=0.2, train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,
                     dnn_type='CNN', reg_schema='Item', maxlen_doc=[maxlen_doc_x, 0], maxlen_sent=[maxlen_sent_x, 0])

    
    '''DRMF-CNNGRU-Item'''
    DRMF(max_iter=max_iter, res_dir=res_dir, lambda_u=1, lambda_v=10, dimension=dimension, vocab_size=vocab_size, init_W=init_W,
                         give_weight=give_weight, DNN_X=DNN_X, DNN_Y=None, emb_dim=emb_dim, num_kernel_per_ws=50,
                         train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,
                         dnn_type='CNN_GRU', reg_schema='Item', maxlen_doc=[maxlen_doc_x, 0], maxlen_sent=[maxlen_sent_x, 0], gru_outdim=50, dropout_rate=0.7)
    
    
    '''for users'''
    DNN_Y = D_all['Y_sequence2D']
    len_doc_y = [len(profile) for profile in DNN_Y]
    len_sent_y = []
    for profile in DNN_Y:
        for review in profile:
            len_sent_y.append(len(review))
            
    # histogram('USER: Number of sequences',len_doc_y)         
    # histogram('USER: Length of sequences',len_sent_y)
    len_doc_y = sorted(len_doc_y)
    len_sent_y = sorted(len_sent_y)
    maxlen_doc_y = len_doc_y[(int)(len(len_doc_y) * threshold_doclen)]
    maxlen_sent_y = len_sent_y[(int)(len(len_sent_y) * threshold_sentlen)]
    print ("Y_sequence2D, maxlen_doc:%d maxlen_sent:%d" % (maxlen_doc_y, maxlen_sent_y))
    DNN_Y = index_to_input(DNN_Y, maxlen_doc_y, maxlen_sent_y)
 
 
    '''DRMF-CNNGRU-User'''
    DRMF(max_iter=max_iter, res_dir=res_dir, lambda_u=10, lambda_v=0.1, dimension=dimension, vocab_size=vocab_size, init_W=init_W,
                      give_weight=give_weight, DNN_X=None, DNN_Y=DNN_Y, emb_dim=emb_dim, num_kernel_per_ws=50,
                      train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,
                      dnn_type='CNN_GRU', reg_schema='User', maxlen_doc=[0, maxlen_doc_y], maxlen_sent=[0, maxlen_sent_y], gru_outdim=50, dropout_rate=0.7)
    
    '''DRMF-CNNGRU-Dual'''
    DRMF(max_iter=max_iter, res_dir=res_dir, lambda_u=100, lambda_v=10, dimension=dimension, vocab_size=vocab_size, init_W=init_W,
                     give_weight=give_weight, DNN_X=DNN_X, DNN_Y=DNN_Y, emb_dim=emb_dim, num_kernel_per_ws=50,
                     train_user=train_user, train_item=train_item, valid_user=valid_user, test_user=test_user, R=R,
                     dnn_type='CNN_GRU', reg_schema='Dual', maxlen_doc=[maxlen_doc_x, maxlen_doc_y], maxlen_sent=[maxlen_sent_x, maxlen_sent_y], gru_outdim=50, dropout_rate=0.7)
                  
