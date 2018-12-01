#++++++++++++++++++++++++++++++++++#
#   Paper                          #	
#++++++++++++++++++++++++++++++++++#

Hao Wu, Zhengxin Zhang, Kun Yue, Binbin Zhang, Jun He, Liangchen Sun. Dual-regularized matrix factorization with deep neural networks for recommender systems. Knowledge-Based Systems, 2018, 145(C):46-58.

#++++++++++++++++++++++++++++++++++#
#   Requirements 				 #	
#++++++++++++++++++++++++++++++++++#

The code is implemented based on ConvMF (http://dm.postech.ac.kr/~cartopy/ConvMF/) and upgraded to exploit the latest libraries of Python, Keras and Tensorflow.

Python >=3.5
Keras >=2.0
Tensorflow >=1.2.0
GPU Support(optional)


#++++++++++++++++++++++++++++++++++#
#   Options for command line  	  #	
#++++++++++++++++++++++++++++++++++#
-c	--do_preprocess					type=bool, help="True or False to preprocess raw data for DRMF (default = False)"
-r	--raw_rating_data_path			type=str, help="Path to raw rating data. data format - user id::item id::rating"
-i	--raw_item_document_data_path	type=str, help="Path to raw item document data. item document consists of multiple text. data format - item id::text1|text2..."
-u	--raw_user_profile_data_path	type=str, help="Path to raw user profile data. user profile consists of multiple text. data format - user id::text1|text2..."
-td	--threshold_length_document		type=float, help="Threshold to control the number of sequences for each user/item (default = 0.5)"
-ts	--threshold_length_sentence		type=float, help="Threshold to control the length of sentence for each user/item (default = 0.8)"
-m	--min_rating					type=int, help="Users who have less than \"min_rating\" ratings will be removed (default = 1)"
-l	--max_length_document			type=float, help="Maximum length of documents for preprocess (default = 200)"
-f	--max_df						type=float, help="Threshold to ignore terms that have a document frequency higher than the given value (default = 0.5)"
-s	--vocab_size					type=int, help="Size of vocabulary (default = 8000)"
-t	--split_ratio					type=float, help="Ratio: 1-ratio, ratio/2 and ratio/2 of the entire dataset (R) will be training, valid and test set, respectively (default = 0.2)"
-d	--data_path						type=str, help="Path to training, valid and test data sets"
-a	--aux_path						type=str, help="Path to R, D_all, sets"
-o	--res_dir						type=str, help="Path to DRMF's result"
-e	--emb_dim						type=int, help="Size of latent dimension for word vectors (default: 200)"
-p	--pretrain_w2v					type=str, help="Path to pretrain word embedding model  to initialize word vectors"
-g	--give_weight					type=bool,help="True or False to give item/user weight of DRMF (default = False)"
-k	--dimension						type=int, help="Size of latent dimension for users and items (default: 50)"
-lu	--lambda_u						type=float,help="Value of user regularizer"
-lv	--lambda_v						type=float,help="Value of item regularizer"
-n	--max_iter						type=int, help="Value of max iteration (default: 200)"
-w	--num_kernel_per_ws				type=int, help="Number of kernels per window size for DNN module (default: 100)"
-b	--binary_rating					type=bool,help="True or False to binarize ratings (default = False)"

#++++++++++++++++++++++++++++++++++#
#   Training & Test 	           #	
#++++++++++++++++++++++++++++++++++#

(1) Downloading datasets
You can download the Yelp dataset from: https://www.yelp.com/dataset/challenge and the Amazon datasets from :http://jmcauley.ucsd.edu/data/amazon/.
For example, download Amazon_Instant_Video_5.json  from http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz  and place it in the path: "data\\Amazon_Instant_Video_5.json".

(2) Preprocessing datasets
Run the script of "reviews_processor.py" to generate user_content.dat, item_content.dat and ratings.dat in the directory of "data\\Amazon_Instant_Video_5", 
or Run the script of "preprocess(AIV).bat" to generate all required datasets for training, validation and test.

(3) Running demos
Run the script of "run(AIV).bat " for rating predictions and  "run_binary(AIV)" for top-n recommendations. 
You should have glove.6B.200d.txt in the directory "./data/glove/", which can be downloaded from http://nlp.stanford.edu/data/glove.6B.zip .
If you want to train the embedding layer of deep neural networks, you should let trainable=True (see CNN_module and CNN_GRU_module in text_analysis/models).

ps: You need create  *.bat files to ease the use of the scripts under Windows or other command files under Linux or Mac OS. 
    
