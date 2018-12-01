'''
Created on Dec 9, 2015
@author: donghyun

Modified on June 20, 2017 
@author: Hao Wu, Zhengxin Zhang

'''
import argparse
import sys
from data_manager import Data_Factory
from util import index_to_input, histogram
from rating_models import DRMF, PMF

parser = argparse.ArgumentParser()

# Option for pre-processing data
parser.add_argument("-c", "--do_preprocess", type=bool,
                    help="True or False to preprocess raw data for DRMF (default = False)", default=False)
parser.add_argument("-r", "--raw_rating_data_path", type=str,
                    help="Path to raw rating data. data format - user id::item id::rating")
parser.add_argument("-i", "--raw_item_document_data_path", type=str,
                    help="Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2...")
##########################################################################################################################################
parser.add_argument("-u", "--raw_user_profile_data_path", type=str,
                    help="Path to raw user profile data. user profile consists of multiple text. data format - user id::text1|text2...")
parser.add_argument("-td", "--threshold_length_document", type=float,
                    help="Threshold to control the number of sequences for each user/item (default = 0.5)", default=0.5)
parser.add_argument("-ts", "--threshold_length_sentence", type=float,
                    help="Threshold to control the length of sentence for each user/item (default = 0.8)", default=0.8)
##########################################################################################################################################

parser.add_argument("-m", "--min_rating", type=int,
                    help="Users who have less than \"min_rating\" ratings will be removed (default = 1)", default=1)
parser.add_argument("-l", "--max_length_document", type=float,
                    help="Maximum length of documents for preprocess (default = 200)", default=200)

parser.add_argument("-f", "--max_df", type=float,
                    help="Threshold to ignore terms that have a document frequency higher than the given value (default = 0.5)", default=0.5)
parser.add_argument("-s", "--vocab_size", type=int,
                    help="Size of vocabulary (default = 8000)", default=8000)
parser.add_argument("-t", "--split_ratio", type=float,
                    help="Ratio: 1-ratio, ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively (default = 0.2)", default=0.2)

# Option for pre-processing data and running DRMF
parser.add_argument("-d", "--data_path", type=str,
                    help="Path to training, valid and test data sets")
parser.add_argument("-a", "--aux_path", type=str, help="Path to R, D_all, sets")

# Option for running DRMF
parser.add_argument("-o", "--res_dir", type=str,
                    help="Path to DRMF's result")
parser.add_argument("-e", "--emb_dim", type=int,
                    help="Size of latent dimension for word vectors (default: 200)", default=200)
parser.add_argument("-p", "--pretrain_w2v", type=str,
                    help="Path to pretrain word embedding model  to initialize word vectors")
parser.add_argument("-g", "--give_weight", type=bool,
                    help="True or False to give item/user weight of DRMF (default = False)", default=True)
parser.add_argument("-k", "--dimension", type=int,
                    help="Size of latent dimension for users and items (default: 50)", default=50)
parser.add_argument("-lu", "--lambda_u", type=float,
                    help="Value of user regularizer", default=0.1)
parser.add_argument("-lv", "--lambda_v", type=float,
                    help="Value of item regularizer", default=0.1)
parser.add_argument("-n", "--max_iter", type=int,
                    help="Value of max iteration (default: 200)", default=200)
parser.add_argument("-w", "--num_kernel_per_ws", type=int,
                    help="Number of kernels per window size for DNN module (default: 100)", default=100)
parser.add_argument("-b", "--binary_rating", type=bool,
                    help="True or False to binarize ratings (default = False)", default=False)

args = parser.parse_args()
do_preprocess = args.do_preprocess
data_path = args.data_path
aux_path = args.aux_path
if data_path is None:
    sys.exit("Argument missing - data_path is required")
if aux_path is None:
    sys.exit("Argument missing - aux_path is required")

data_factory = Data_Factory()

if do_preprocess:
    path_rating = args.raw_rating_data_path
    path_itemtext = args.raw_item_document_data_path
    path_usertext = args.raw_user_profile_data_path
    min_rating = args.min_rating
    max_length = args.max_length_document
    max_df = args.max_df
    vocab_size = args.vocab_size
    split_ratio = args.split_ratio

    print ("=================================Preprocess Option Setting=================================")
    print ("\tsaving preprocessed aux path - %s" % aux_path)
    print ("\tsaving preprocessed data path - %s" % data_path)
    print ("\trating data path - %s" % path_rating)
    print ("\tdocument data path - %s" % path_itemtext)
    print ("\tprofile data path - %s" % path_usertext)
    print ("\tmin_rating: %d\n\tmax_length_document: %d\n\tmax_df: %.1f\n\tvocab_size: %d\n\tsplit_ratio: %.1f" \
        % (min_rating, max_length, max_df, vocab_size, split_ratio))
    print ("===========================================================================================")

    R, D_all = data_factory.preprocess_ext(path_rating, path_itemtext, path_usertext, min_rating, max_length, max_df, vocab_size)
    data_factory.save(aux_path, R, D_all)
    data_factory.generate_train_valid_test_file_from_R(data_path, R, split_ratio)
else:
    res_dir = args.res_dir
    emb_dim = args.emb_dim
    pretrain_w2v = args.pretrain_w2v
    dimension = args.dimension
    lambda_u = args.lambda_u
    lambda_v = args.lambda_v
    max_iter = args.max_iter
    num_kernel_per_ws = args.num_kernel_per_ws
    give_weight = args.give_weight
    threshold_doclen = args.threshold_length_document
    threshold_sentlen = args.threshold_length_sentence
    binary_rating =args.binary_rating
    
    if res_dir is None:
        sys.exit("Argument missing - res_dir is required")
    if lambda_u is None:
        sys.exit("Argument missing - lambda_u is required")
    if lambda_v is None:
        sys.exit("Argument missing - lambda_v is required")

    print ("===================================MF Option Setting===================================")
    print ("\tbinarizing ratings - %s" % binary_rating)
    print ("\tdata path - %s" % data_path)
    print ("\tresult path - %s" % res_dir)
    print ("\tpretrained w2v data path - %s" % pretrain_w2v)
    print ("\tdimension: %d\n\tlambda_u: %.4f\n\tlambda_v: %.4f\n\tmax_iter: %d\n\tnum_kernel_per_ws: %d" \
        % (dimension, lambda_u, lambda_v, max_iter, num_kernel_per_ws))
    print ("===========================================================================================")
    R, D_all = data_factory.load(aux_path,binary_rating)
    train_user = data_factory.read_rating(data_path + '/train_user.dat',binary_rating)
    train_item = data_factory.read_rating(data_path + '/train_item.dat',binary_rating)
    valid_user = data_factory.read_rating(data_path + '/valid_user.dat',binary_rating)
    test_user = data_factory.read_rating(data_path + '/test_user.dat',binary_rating)

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
                  

