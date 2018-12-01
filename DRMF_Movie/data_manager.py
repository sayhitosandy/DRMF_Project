'''
Created on Nov 9, 2015
@author: donghyun

Modified on March 20, 2017 
@author: Hao Wu, Zhengxin Zhang
'''

from operator import itemgetter
import os
import pickle
import random
import sys

from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import numpy as np


class Data_Factory():

    def load(self, path, binary_rating=False):
        R = pickle.load(open(path + "/ratings.all", "rb"))
        print ("Load preprocessed rating data - %s" % (path + "/ratings.all"))
        if binary_rating:
            R=R.power(0)
        D_all = pickle.load(open(path + "/document.all", "rb"))
        print ("Load preprocessed document data - %s" % (path + "/document.all"))
        return R, D_all

    def save(self, path, R, D_all):
        if not os.path.exists(path):
            os.makedirs(path)
        print ("Saving preprocessed rating data - %s" % (path + "/ratings.all"))
        pickle.dump(R, open(path + "/ratings.all", "wb"))
        print ("Done!")
        print ("Saving preprocessed document data - %s" % (path + "/document.all"))
        pickle.dump(D_all, open(path + "/document.all", "wb"))
        print ("Done!")

    def read_rating(self, path, binary_rating=False):
        results = []
        if os.path.isfile(path):
            raw_ratings = open(path, 'r')
        else:
            print ("Path (preprocessed) is wrong!")
            sys.exit()
        index_list = []
        rating_list = []
        all_line = raw_ratings.read().splitlines()
        for line in all_line:
            tmp = line.split()
            num_rating = int(tmp[0])
            if num_rating > 0:
                tmp_i, tmp_r = zip(*(elem.split(":") for elem in tmp[1::]))
                index_list.append(np.array(tmp_i, dtype=int))
                if not binary_rating:
                    rating_list.append(np.array(tmp_r, dtype=float))
                else:
                    rating_list.append(np.ones(len(tmp_r),dtype=float))
                    
            else:
                index_list.append(np.array([], dtype=int))
                rating_list.append(np.array([], dtype=float))

        results.append(index_list)
        results.append(rating_list)
        return results
        

    def read_pretrained_word2vec(self, path, vocab, dim):
        if os.path.isfile(path):
            raw_word2vec = open(path, 'r', encoding='UTF-8')
        else:
            print ("Path (word2vec) is wrong!")
            sys.exit()

        word2vec_dic = {}
        all_line = raw_word2vec.read().splitlines()
        mean = np.zeros(dim)
        count = 0
        for line in all_line:
            tmp = line.split()
            _word = tmp[0]
            _vec = np.array(tmp[1:], dtype=float)
            if _vec.shape[0] != dim:
                print ("Mismatch the dimension of pre-trained word vector with word embedding dimension!")
                sys.exit()
            word2vec_dic[_word] = _vec
            mean = mean + _vec
            count = count + 1

        mean = mean / count

        W = np.zeros((len(vocab) + 1, dim))
        count = 0
        for _word, i in vocab:
            if _word in word2vec_dic:
                W[i + 1] = word2vec_dic[_word]
                count = count + 1
            else:
                W[i + 1] = np.random.normal(mean, 0.1, size=dim)

        print ("%d words exist in the given pretrained model" % count)

        return W

    def split_data(self, ratio, R):
        print ("Randomly splitting rating data into training set (%.1f) and test set (%.1f)..." % (1 - ratio, ratio))
        train = []
        for i in range(R.shape[0]):
            user_rating = R[i].nonzero()[1]
            np.random.shuffle(user_rating)
            try:
                train.append((i, user_rating[0]))
            except:
                p=1

        remain_item = set(range(R.shape[1])) - set(list(zip(*train))[1])

        for j in remain_item:
            item_rating = R.tocsc().T[j].nonzero()[1]
            np.random.shuffle(item_rating)
            try:
                train.append((item_rating[0], j))
            except:
                continue

        rating_list = set(zip(R.nonzero()[0], R.nonzero()[1]))
        total_size = len(rating_list)
        remain_rating_list = list(rating_list - set(train))
        random.shuffle(remain_rating_list)

        num_addition = int((1 - ratio) * total_size) - len(train)
        if num_addition < 0:
            print ('this ratio cannot be handled')
            sys.exit()
        else:
            train.extend(remain_rating_list[:num_addition])
            tmp_test = remain_rating_list[num_addition:]
            random.shuffle(tmp_test)
            valid = tmp_test[::2]
            test = tmp_test[1::2]
            trainset_u_idx, trainset_i_idx = zip(*train)
            trainset_u_idx = set(trainset_u_idx)
            trainset_i_idx = set(trainset_i_idx)
            print(len(trainset_u_idx))
            print(R.shape[0])
            print(len(trainset_i_idx))
            print(R.shape[1])
            if len(trainset_u_idx) != R.shape[0] or len(trainset_i_idx) != R.shape[1]:
                print ("Fatal error in split function. Check your data again or contact authors")
                sys.exit()

        print ("Finish constructing training set and test set")
        return train, valid, test

    def generate_train_valid_test_file_from_R(self, path, R, ratio):
        '''
        Split randomly rating matrix into training set, valid set and test set with given ratio (valid+test)
        and save three data sets to given path.
        Note that the training set contains at least a rating on every user and item.

        Input:
        - path: path to save training set, valid set, test set
        - R: rating matrix (csr_matrix)
        - ratio: (1-ratio), ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively
        '''
        train, valid, test = self.split_data(ratio, R)
        print ("Save training set and test set to %s..." % path)
        if not os.path.exists(path):
            os.makedirs(path)

        R_lil = R.tolil()
        user_ratings_train = {}
        item_ratings_train = {}
        for i, j in train:
            if i in user_ratings_train:
                user_ratings_train[i].append(j)
            else:
                user_ratings_train[i] = [j]

            if j in item_ratings_train:
                item_ratings_train[j].append(i)
            else:
                item_ratings_train[j] = [i]

        user_ratings_valid = {}
        item_ratings_valid = {}
        for i, j in valid:
            if i in user_ratings_valid:
                user_ratings_valid[i].append(j)
            else:
                user_ratings_valid[i] = [j]

            if j in item_ratings_valid:
                item_ratings_valid[j].append(i)
            else:
                item_ratings_valid[j] = [i]

        user_ratings_test = {}
        item_ratings_test = {}
        for i, j in test:
            if i in user_ratings_test:
                user_ratings_test[i].append(j)
            else:
                user_ratings_test[i] = [j]

            if j in item_ratings_test:
                item_ratings_test[j].append(i)
            else:
                item_ratings_test[j] = [i]

        f_train_user = open(path + "/train_user.dat", "w")
        f_valid_user = open(path + "/valid_user.dat", "w")
        f_test_user = open(path + "/test_user.dat", "w")

        formatted_user_train = []
        formatted_user_valid = []
        formatted_user_test = []

        for i in range(R.shape[0]):
            if i in user_ratings_train:
                formatted = [str(len(user_ratings_train[i]))]
                formatted.extend(["%d:%.1f" % (j, R_lil[i, j])
                                  for j in sorted(user_ratings_train[i])])
                formatted_user_train.append(" ".join(formatted))
            else:
                formatted_user_train.append("0")

            if i in user_ratings_valid:
                formatted = [str(len(user_ratings_valid[i]))]
                formatted.extend(["%d:%.1f" % (j, R_lil[i, j])
                                  for j in sorted(user_ratings_valid[i])])
                formatted_user_valid.append(" ".join(formatted))
            else:
                formatted_user_valid.append("0")

            if i in user_ratings_test:
                formatted = [str(len(user_ratings_test[i]))]
                formatted.extend(["%d:%.1f" % (j, R_lil[i, j])
                                  for j in sorted(user_ratings_test[i])])
                formatted_user_test.append(" ".join(formatted))
            else:
                formatted_user_test.append("0")

        f_train_user.write("\n".join(formatted_user_train))
        f_valid_user.write("\n".join(formatted_user_valid))
        f_test_user.write("\n".join(formatted_user_test))

        f_train_user.close()
        f_valid_user.close()
        f_test_user.close()
        print ("\ttrain_user.dat, valid_user.dat, test_user.dat files are generated.")

        f_train_item = open(path + "/train_item.dat", "w")
        f_valid_item = open(path + "/valid_item.dat", "w")
        f_test_item = open(path + "/test_item.dat", "w")

        formatted_item_train = []
        formatted_item_valid = []
        formatted_item_test = []

        for j in range(R.shape[1]):
            if j in item_ratings_train:
                formatted = [str(len(item_ratings_train[j]))]
                formatted.extend(["%d:%.1f" % (i, R_lil[i, j])
                                  for i in sorted(item_ratings_train[j])])
                formatted_item_train.append(" ".join(formatted))
            else:
                formatted_item_train.append("0")

            if j in item_ratings_valid:
                formatted = [str(len(item_ratings_valid[j]))]
                formatted.extend(["%d:%.1f" % (i, R_lil[i, j])
                                  for i in sorted(item_ratings_valid[j])])
                formatted_item_valid.append(" ".join(formatted))
            else:
                formatted_item_valid.append("0")

            if j in item_ratings_test:
                formatted = [str(len(item_ratings_test[j]))]
                formatted.extend(["%d:%.1f" % (i, R_lil[i, j])
                                  for i in sorted(item_ratings_test[j])])
                formatted_item_test.append(" ".join(formatted))
            else:
                formatted_item_test.append("0")

        f_train_item.write("\n".join(formatted_item_train))
        f_valid_item.write("\n".join(formatted_item_valid))
        f_test_item.write("\n".join(formatted_item_test))

        f_train_item.close()
        f_valid_item.close()
        f_test_item.close()
        print ("\ttrain_item.dat, valid_item.dat, test_item.dat files are generated.")

        print ("Done!")

    def generate_CTRCDLformat_content_file_from_D_all(self, path, D_all):
        '''
        Write word index with word count in document for CTR&CDL experiment

        '''
        f_text = open(path + "mult.dat", "w")
        X = D_all['X_base']
        formatted_text = []
        for i in range(X.shape[0]):
            word_count = sorted(set(X[i].nonzero()[1]))
            formatted = [str(len(word_count))]
            formatted.extend(["%d:%d" % (j, X[i, j]) for j in word_count])
            formatted_text.append(" ".join(formatted))

        f_text.write("\n".join(formatted_text))
        f_text.close()

    def preprocess(self, path_rating, path_itemtext, path_usertext, min_rating,
                   _max_length, _max_df, _vocab_size):
        '''
        Preprocess rating and document data.

        Input:
            - path_rating: path for rating data (data format - user_id::item_id::rating)
            - path_itemtext: path for review or synopsis data (data format - item_id::text1|text2|text3|....)
            - path_usertext: path for review or synopsis data (data format - user_id::text1|text2|text3|....)
            - min_rating: users who have less than "min_rating" ratings will be removed (default = 1)
            - _max_length: maximum length of document of each item (default = 300)
            - _max_df: terms will be ignored that have a document frequency higher than the given threshold (default = 0.5)
            - vocab_size: vocabulary size (default = 8000)

        Output:
            - R: rating matrix (csr_matrix: row - user, column - item)
            - D_all['X_sequence']: list of sequence of word index of each item ([[1,2,3,4,..],[2,3,4,...],...])
            - D_all['X_vocab']: list of tuple (word, index) in the given corpus
            - D_all['Y_sequence']: list of sequence of word index of each user ([[1,2,3,4,..],[2,3,4,...],...])
            - D_all['Y_vocab']: list of tuple (word, index) in the given corpus
        '''
        # Validate data paths
        if os.path.isfile(path_rating):
            raw_ratings = open(path_rating, 'r')
            print ("Path - rating data: %s" % path_rating)
        else:
            print ("Path(rating) is wrong!")
            sys.exit()

        if os.path.isfile(path_itemtext):
            raw_content = open(path_itemtext, 'r')
            print ("Path - document data: %s" % path_itemtext)
        else:
            print ("Path(item text) is wrong!")
            sys.exit()
            
        ##############################################################   
        if os.path.isfile(path_usertext):
            raw_profile = open(path_usertext, 'r')
            print ("Path - profile data: %s" % path_usertext)
        else:
            print ("Path(user text) is wrong!")
            sys.exit()
        ##############################################################

        # 1st scan document file to filter items which have documents
        tmp_iid_plot = set()
        all_line = raw_content.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            i = tmp[0]
            tmp_plot = tmp[1].split('|')
            if tmp_plot[0] == '':
                continue
            tmp_iid_plot.add(i)
        raw_content.close()
        # 1st scan document file to filter users which have documents
        ##############################################################
        tmp_uid_plot = set()
        all_line = raw_profile.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            i = tmp[0]
            tmp_plot = tmp[1].split('|')
            if tmp_plot[0] == '':
                continue
            tmp_uid_plot.add(i)
        raw_profile.close()
        #############################################################

        print ("Preprocessing rating data...")
        print ("\tCounting # ratings of each user and removing users having less than %d ratings..." % min_rating)
        # 1st scan rating file to check # ratings of each user
        all_line = raw_ratings.read().splitlines()
        tmp_user = {}
        for line in all_line:
            tmp = line.split('::')
            u = tmp[0]
            i = tmp[1]
            if (i in tmp_iid_plot):
                if (u not in tmp_user):
                    tmp_user[u] = 1
                else:
                    tmp_user[u] = tmp_user[u] + 1

        raw_ratings.close()

        # 2nd scan rating file to make matrix indices of users and items
        # with removing users and items which are not satisfied with the given
        # condition
        raw_ratings = open(path_rating, 'r')
        all_line = raw_ratings.read().splitlines()
        userset = {}
        itemset = {}
        user_idx = 0
        item_idx = 0

        user = []
        item = []
        rating = []

        for line in all_line:
            tmp = line.split('::')
            u = tmp[0]
            if u not in tmp_user:
                continue
            i = tmp[1]
            # An user will be skipped where the number of ratings of the user
            # is less than min_rating.
            if tmp_user[u] >= min_rating:
                #####################################################
                if (u not in userset) and (u not in tmp_uid_plot):
                #####################################################
                    userset[u] = user_idx
                    user_idx = user_idx + 1

                if (i not in itemset) and (i in tmp_iid_plot):
                    itemset[i] = item_idx
                    item_idx = item_idx + 1
            else:
                continue

            if u in userset and i in itemset:
                u_idx = userset[u]
                i_idx = itemset[i]

                user.append(u_idx)
                item.append(i_idx)
                rating.append(float(tmp[2]))
                
        raw_ratings.close()

        R = csr_matrix((rating, (user, item)))

        print ("Finish preprocessing rating data - # user: %d, # item: %d, # ratings: %d" % (R.shape[0], R.shape[1], R.nnz))

        # 2nd scan document file to make idx2plot dictionary according to
        # indices of items in rating matrix
        print ("Preprocessing item document...")

        # Read Document File
        raw_content = open(path_itemtext, 'r')
        max_length = _max_length
        map_idtoplot = {}
        all_line = raw_content.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            if tmp[0] in itemset:
                i = itemset[tmp[0]]
                tmp_plot = tmp[1].split('|')
                eachid_plot = (' '.join(tmp_plot)).split()[:max_length]
                map_idtoplot[i] = ' '.join(eachid_plot)

        print ("\tRemoving stop words...")
        print ("\tFiltering words by TF-IDF score with max_df: %.1f, vocab_size: %d" % (_max_df, _vocab_size))

        # Make vocabulary by document
        vectorizer = TfidfVectorizer(max_df=_max_df, stop_words={
                                     'english'}, max_features=_vocab_size)
        Raw_X = [map_idtoplot[_i] for _i in range(R.shape[1])]
        vectorizer.fit(Raw_X)
        vocab = vectorizer.vocabulary_
        X_vocab = sorted(vocab.items(), key=itemgetter(1))

        # Make input for run1
        X_sequence = []
        for i in range(R.shape[1]):
            X_sequence.append(
                [vocab[word] + 1 for word in map_idtoplot[i].split() if word in vocab])

        '''Make input for CTR & CDL'''
        baseline_vectorizer = CountVectorizer(vocabulary=vocab)
        X_base = baseline_vectorizer.fit_transform(Raw_X)

        ###################################################################
        # 2nd scan profile file to make idx2plot dictionary according to
        # indices of users in rating matrix
        print ("Preprocessing user profile...")

        # Read Document File
        raw_profile = open(path_usertext, 'r')
        max_length = _max_length
        map_idtoplot = {}
        all_line = raw_profile.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            if tmp[0] in userset:
                u = userset[tmp[0]]
                tmp_plot = tmp[1].split('|')
                eachid_plot = (' '.join(tmp_plot)).split()[:max_length]
                map_idtoplot[u] = ' '.join(eachid_plot)

        print ("\tRemoving stop words...")
        print ("\tFiltering words by TF-IDF score with max_df: %.1f, vocab_size: %d" % (_max_df, _vocab_size))

        # Make vocabulary by document
        vectorizer = TfidfVectorizer(max_df=_max_df, stop_words={
                                     'english'}, max_features=_vocab_size)
        Raw_Y = [map_idtoplot[_u] for _u in range(R.shape[0])]
        vectorizer.fit(Raw_Y)
        vocab = vectorizer.vocabulary_
        Y_vocab = sorted(vocab.items(), key=itemgetter(1))

        # Make input for run1
        Y_sequence = []
        for u in range(R.shape[0]):
            Y_sequence.append(
                [vocab[word] + 1 for word in map_idtoplot[u].split() if word in vocab])

        '''Make input for CTR & CDL'''
        baseline_vectorizer = CountVectorizer(vocabulary=vocab)
        Y_base = baseline_vectorizer.fit_transform(Raw_Y)

        D_all = {
            'X_sequence': X_sequence,
            'X_base': X_base,
            'X_vocab': X_vocab,
            'Y_sequence': Y_sequence,
            'Y_base': Y_base,
            'Y_vocab': Y_vocab,
        }

        print ("Finish preprocessing document data!")
        
        ###########################################
        
        return R, D_all
    
    def preprocess_ext(self, path_rating, path_itemtext, path_usertext, min_rating,
                   _max_length, _max_df, _vocab_size):
        '''
        Preprocess rating and document data.
        Input:
            - path_rating: path for rating data (data format - user_id::item_id::rating)
            - path_itemtext: path for review or synopsis data (data format - item_id::text1|text2|text3|....)
            - path_usertext: path for review or synopsis data (data format - user_id::text1|text2|text3|....)
            - min_rating: users who have less than "min_rating" ratings will be removed (default = 1)
            - _max_length: maximum length of document of each item (default = 300)
            - _max_df: terms will be ignored that have a document frequency higher than the given threshold (default = 0.5)
            - vocab_size: vocabulary size (default = 8000)
        Output:
            - R: rating matrix (csr_matrix: row - user, column - item)
            - D_all['X_sequence2D']: list of sequence of word index of each item ([[1,2,3,4,..],[2,3,4,...],...])
            - D_all['Y_sequence2D']: list of sequence of word index of each user ([[1,2,3,4,..],[2,3,4,...],...])
            - D_all['XY_vocab']: list of tuple (word, index) in the given corpus
        '''
        # Validate data paths
        if os.path.isfile(path_rating):
            raw_ratings = open(path_rating, 'r')
            print ("Path - rating data: %s" % path_rating)
        else:
            print ("Path(rating) is wrong!")
            sys.exit()

        if os.path.isfile(path_itemtext):
            raw_content = open(path_itemtext, 'r')
            print ("Path - document data: %s" % path_itemtext)
        else:
            print ("Path(item text) is wrong!")
            sys.exit()
            
        ##############################################################   
        if os.path.isfile(path_usertext):
            raw_profile = open(path_usertext, 'r')
            print ("Path - profile data: %s" % path_usertext)
        else:
            print ("Path(user text) is wrong!")
            sys.exit()
        ##############################################################

        # 1st scan document file to filter items which have documents
        tmp_iid_plot = set()
        all_line = raw_content.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            if len(tmp)==1:
                continue
            i = tmp[0]
            tmp_plot = tmp[1].split('|')
            if tmp_plot[0] == '':
                continue
            tmp_iid_plot.add(i)
        raw_content.close()
        # print tmp_iid_plot
        # 1st scan document file to filter users which have documents
        ##############################################################
        tmp_uid_plot = set()
        all_line = raw_profile.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            if len(tmp)==1:
                continue
            i = tmp[0]
            tmp_plot = tmp[1].split('|')
            if tmp_plot[0] == '':
                continue
            tmp_uid_plot.add(i)
        raw_profile.close()
        # print tmp_uid_plot
        #############################################################

        print ("Preprocessing rating data...")
        print ("\tCounting # ratings of each user and removing users having less than %d ratings..." % min_rating)
        # 1st scan rating file to check # ratings of each user
        all_line = raw_ratings.read().splitlines()
        tmp_user = {}
        for line in all_line:
            tmp = line.split('::')
            u = tmp[0]
            i = tmp[1]
            if (i in tmp_iid_plot):
                if (u not in tmp_user):
                    tmp_user[u] = 1
                else:
                    tmp_user[u] = tmp_user[u] + 1

        raw_ratings.close()

        # 2nd scan rating file to make matrix indices of users and items
        # with removing users and items which are not satisfied with the given
        # condition
        raw_ratings = open(path_rating, 'r')
        all_line = raw_ratings.read().splitlines()
        userset = {}
        itemset = {}
        user_idx = 0
        item_idx = 0

        user = []
        item = []
        rating = []

        for line in all_line:
            tmp = line.split('::')
            # print tmp
            u = tmp[0]
            if u not in tmp_user:
                continue
            i = tmp[1]
            # An user will be skipped where the number of ratings of the user
            # is less than min_rating.
            if tmp_user[u] >= min_rating:
                #####################################################
                if (u not in userset) and (u in tmp_uid_plot):
                #####################################################
                    userset[u] = user_idx
                    user_idx = user_idx + 1

                if (i not in itemset) and (i in tmp_iid_plot):
                    itemset[i] = item_idx
                    item_idx = item_idx + 1
            else:
                continue

            if u in userset and i in itemset:
                u_idx = userset[u]
                i_idx = itemset[i]

                user.append(u_idx)
                item.append(i_idx)
                rating.append(float(tmp[2]))

        raw_ratings.close()
       
        R = csr_matrix((rating, (user, item)))

        print ("Finish preprocessing rating data - # user: %d, # item: %d, # ratings: %d" % (R.shape[0], R.shape[1], R.nnz))

        # 2nd scan document file to make idx2plot dictionary according to
        # indices of items in rating matrix
        print ("Preprocessing item document...")

        # Read Document File
        raw_content = open(path_itemtext, 'r')
        max_length = _max_length
        map_idtoplots_x = {}
        all_line = raw_content.read().splitlines()
        # print all_line
        for line in all_line:
            tmp = line.split('::')
            if tmp[0] in itemset:
                i = itemset[tmp[0]]
                tmp_plot = tmp[1].split('|')
                map_idtoplots_x[i] = [tmp_plot[k].split()[:max_length] for k in range(len(tmp_plot))]
        # print "map_idtoplots_x: \n %s" % map_idtoplots_x
        print ("\tRemoving stop words...")
        print ("\tFiltering words by TF-IDF score with max_df: %.1f, vocab_size: %d" % (_max_df, _vocab_size))

        # Raw_X = [map_idtoplots_x[_i] for _i in range(R.shape[1])]
        Raw_X = []
        for i in range(R.shape[1]):
            tmp_string = []
            for k in range(len(map_idtoplots_x[i])):
                tmp_string.append(' '.join(map_idtoplots_x[i][k]));
            Raw_X.append(' '.join(tmp_string))
            # print "Raw_X[%d]: \n %s" %(i, Raw_X[i]) 

        ###################################################################
        # 2nd scan profile file to make idx2plot dictionary according to
        # indices of users in rating matrix
        print ("Preprocessing user profile...")

        # Read Document File
        raw_profile = open(path_usertext, 'r')
        max_length = _max_length
        map_idtoplots_y = {}
        all_line = raw_profile.read().splitlines()
        for line in all_line:
            tmp = line.split('::')
            if tmp[0] in userset:
                u = userset[tmp[0]]
                tmp_plot = tmp[1].split('|')
                map_idtoplots_y[u] = [tmp_plot[k].split()[:max_length] for k in range(len(tmp_plot))]
        
        # print "map_idtoplots_y: \n %s" % map_idtoplots_y

        print ("\tRemoving stop words...")
        print ("\tFiltering words by TF-IDF score with max_df: %.1f, vocab_size: %d" % (_max_df, _vocab_size))

        # Raw_Y = [map_idtoplots_y[_u] for _u in range(R.shape[0])]
        Raw_Y = []
        for u in range(R.shape[0]):
            tmp_string = []
            for k in range(len(map_idtoplots_y[u])):
                tmp_string.append(' '.join(map_idtoplots_y[u][k]));
            Raw_Y.append(' '.join(tmp_string))
            # print "Raw_Y[%d]: \n %s" %(u, Raw_Y[u]) 
        
        Raw_XY = list(set(Raw_X + Raw_Y))
        # print  "Raw_XY: \n %s" %  Raw_XY    
        # Make vocabulary by document
        vectorizer = TfidfVectorizer(max_df=_max_df, stop_words={
                                     'english'}, max_features=_vocab_size)

        vectorizer.fit(Raw_XY)
        vocab = vectorizer.vocabulary_
        XY_vocab = sorted(vocab.items(), key=itemgetter(1))
        # print "size:%d vocab:  \n %s" % (len(vocab),vocab)
        # print "XY_vocab: \n %s" % XY_vocab

        # Make input for run1
        X_sequence2D = []
        for i in range(R.shape[1]):
            X_sequence2D.append(
                [[vocab[word] + 1 for word in map_idtoplots_x[i][k] if word in vocab] for k in range(len(map_idtoplots_x[i]))])
        # print "X_sequence2D: \n %s" % X_sequence2D
                    
        Y_sequence2D = []
        for u in range(R.shape[0]):
            Y_sequence2D.append(
                [[vocab[word] + 1 for word in map_idtoplots_y[u][k] if word in vocab] for k in range(len(map_idtoplots_y[u]))])
        # print "Y_sequence2D: \n %s" % Y_sequence2D
        
        D_all = {
            'X_sequence2D': X_sequence2D,
            'Y_sequence2D': Y_sequence2D,
            'XY_vocab': XY_vocab,
        }
        print ("Finish preprocessing document data!")
        
        ###########################################
        
        return R, D_all