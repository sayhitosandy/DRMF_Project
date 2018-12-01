'''
Created on Dec 9, 2015
@author: donghyun

Modified on June 20, 2017 
@author: Hao Wu, Zhengxin Zhang

'''
import argparse
import sys
from data_manager import Data_Factory

parser = argparse.ArgumentParser()

# Option for pre-processing data
parser.add_argument("-c", "--do_preprocess", type=bool,
                    help="True or False to preprocess raw data for DRMF (default = False)", default=False)
parser.add_argument("-r", "--raw_rating_data_path", type=str,
                    help="Path to raw rating data. data format - user id::item id::rating")
parser.add_argument("-i", "--raw_item_document_data_path", type=str,
                    help="Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2...")
parser.add_argument("-u", "--raw_user_profile_data_path", type=str,
                    help="Path to raw user profile data. user profile consists of multiple text. data format - user id::text1|text2...")
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
parser.add_argument("-d", "--data_path", type=str,
                    help="Path to training, valid and test data sets")
parser.add_argument("-a", "--aux_path", type=str, help="Path to R, D_all, sets")



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
    print ("do nothing...")
