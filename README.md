# DRMF_Project
Experiments on Dual-Regularized Matrix Factorization with Deep Neural Networks for Recommender Systems

### Introduction
In this project, we analyze the results of a recent paper related to the use of deep-learning in collaborating filtering, and experiment their models on popular drug target interaction datasets and our own IIIT-Delhi's database of movies.

There are 2 folders:
1. DRMF_DrugTarget: This folder contains the results obtained from experiments on popular drug target interaction datasets.
2. DRMF_Movie: This folder contains the results obtained from experiments on IIIT-Delhi's database of movies.
3. DRMF_AIV: This folder contains the results obtained from experiments on Amazon Instant Video dataset (used by the authors).

### How to run?
1. Drug Target Datasets: Go to DRMF_DrugTarget.
	To preprocess, run `preprocess.bat`.
	To split the dataset in training, validation and test splits, run `train_test_split.bat`.
	To run the model, run `run.bat`.

2. Amazon Instant Video Dataset: Go to DRMF_AIV.
	To preprocess, run `preprocess(AIV).bat`.
	To run the model, run `run(AIV).bat`.
	Also, make sure you have `glove.6B.200d.txt` in the directory `./data/glove/`, which can be downloaded from http://nlp.stanford.edu/data/glove.6B.zip .

3. IIIT-Delhi's Movies Database: Go to DRMF_Movie.
	The files are already preprocessed and are present in `./test/FS/`.
	To split the dataset in training, validation and test splits, run `python ./run.py -d ./test/FS -a ./test/FS -c True -r ./test/FS/ratings.dat -i ./test/FS/item_content.dat -u ./test/FS/user_content.dat -m 1`.
	To run the model, run `python ./run.py -d ./test/FS -a ./test/FS -o ./outputs/FS -e 200 -p ./data/glove/glove.6B.200d.txt -g False`.

All the outputs are stored in `state.log` files in the respective output folders.

### References

1. Wu  et al., Dual-regularized matrix factorization with deep neural networks for recommender systems, Knowledge-Based Systems, Volume 145 (2018), https://doi.org/10.1016/j.knosys.2018.01.003 .

2. Ruslan Salakhutdinov and Andriy Mnih, Probabilistic Matrix Factorization, NIPS, 2008.

3. Ezzat, A., Computational prediction of drug–target interactions using chemogenomic  approaches: an empirical survey, Brief Bioinformm 2018 Jan 24, https://doi.org/10.1093/bib/bby002 .

4. Ezzat, A., Drug-target interaction prediction via class imbalance-aware ensemble learning, BMC BioinformaticsBMC series – open, inclusive and trusted  2016, 17 (Suppl 19), https://doi.org/10.1186/s12859-016-1377-y .

5. Majumdar, A, FlickScore dataset, http://flickscore.iiitd.edu.in .