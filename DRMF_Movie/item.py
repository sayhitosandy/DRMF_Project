import numpy as np
import pandas as pd
import os
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

nums_list=[1,2,3,4,5]
def mean(numbers):
    count=0
    sumo=0
    for i in numbers:
        if (i>0):
            count+=1
            sumo+=i
    return sumo/count

def find_avg_rating(user, user_movies_matrix):
    return mean(user_movies_matrix.iloc[user].tolist())

def find_top_k_simusers(user, user_similarity_matrix, k):
    list_of_sims=user_similarity_matrix[user].tolist()
    sim_tuples=[]
    for i in range (len(list_of_sims)):
        sim_tuples.append((i, list_of_sims[i]))
    #sorted(range(len(s)), key=lambda k: s[k])
    sim_tuples=sorted(sim_tuples, key=lambda x: x[1], reverse=True)[1:k+1]
    return (sim_tuples)

def predict_rating(user, movie, user_similarity_matrix, user_movies_matrix, k):
    # Find top k similar users
    top_k_neighbours=find_top_k_simusers(user, user_similarity_matrix, k)

    # Get their ratings for movie m
    #print(top_k_neighbours)
    adjusted_weights=[]
    ratings=[]
    sum_of_weights=0
    for i in range(len(top_k_neighbours)):
        sum_of_weights+=top_k_neighbours[i][1]
    for i in range(len(top_k_neighbours)):
        adjusted_weights.append(top_k_neighbours[i][1]/sum_of_weights)
        ratings.append(user_movies_matrix.iloc[top_k_neighbours[i][0]][movie])
    p_rating=find_avg_rating(user, user_movies_matrix)+sum([x*y for x,y in zip(ratings,adjusted_weights)])
    return (p_rating)

avg_mae=0

for i in range(1,2):
    entire_data=pd.read_csv('Divyanshu_Data/blakely_preprocess.csv', sep='\t', header=None)
    user_movies_matrix = [[0 for x in range(1000)] for y in range(31)] 

    for i in range(1,len(entire_data)):
        movie_string=entire_data.iloc[i][0]
        movie_list=movie_string.split(",")
        movie_list=movie_list[1:]

        for j in range(1,len(movie_list)):
            user_movies_matrix[i][j]=float(movie_list[j])
        #user_movies_matrix[i]=float(movie[])
    
    

    # Read train data
    #train_data=pd.read_csv(train, sep='\t', header=None)
    train_data.columns=['User','Movie','Rating','TS']
    train_data=train_data.drop(['TS'], axis=1)
    
    user_movies_matrix=train_data.pivot(index='User', columns='Movie', values='Rating').reset_index(drop=True)
    user_movies_matrix.fillna(0, inplace = True)
    user_movies_matrix=user_movies_matrix.T
    user_similarity_matrix=pd.DataFrame(pairwise_distances(user_movies_matrix.as_matrix(), metric="cosine"))

    test_data=pd.read_csv(test, sep='\t', header=None)
    test_data.columns=['User','Movie','Rating','TS']
    test_data=test_data.drop(['TS'], axis=1)
    
    count=0
    sums=0
    for index,row in test_data.iterrows():
        try:
            movie=row['User']
            user=row['Movie']
            rating=row['Rating']
            count+=1

            p_rating=predict_rating(user, movie, user_similarity_matrix, user_movies_matrix, k)
            #print(p_rating)
            sums+=abs(p_rating-rating)
        except:
            x=1
    print(sums/count)
    avg_mae+=sums/count

print(avg_mae/5)
