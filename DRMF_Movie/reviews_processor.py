'''
Created on May 25, 2018
@author: Hao Wu, haow@ynu.edu.cn
'''

import json 
import os 
from collections import OrderedDict


class  Str2ID():
    '''
    Mapping strings to unique identifiers
    '''

    def __init__(self):
        self.map = {}
        
    def assignID(self, key):
        if key not in self.map:
            self.map[key] = len(self.map)
        return self.map[key]
    
    def getID(self, key):
        return self.map.get(key, -1)
    
    def dump(self, mapFile):
        with open(mapFile, 'w') as f:
            f.write(str(len(self.map)) + '\n')
            for k, v in self.map.items(): 
                f.write('\t'.join([str(k), str(v)]) + '\n')

    def size(self):
        return len(self.map)

                
class ReviewProcessor():
    
    def writeRawContent(self, outDat, reviews_with_users_items, saveMarks):
        '''
        :param outDat: path for stored dataset 
        :param reviews_with_users_items: an ordered dictionary to aggregate the reviews associated with each user or item
        :param saveMarks: there will be a mark (like [1375574400,719], the first is timestamp, the second is a userID or an itemID) placed on each review.
        '''
        with open(outDat, 'w') as f:
            for user_item_id in reviews_with_users_items:
                f.write(str(user_item_id) + "::")
                reviews_ordered = sorted(reviews_with_users_items[user_item_id].items()) 
                for k, v in reviews_ordered:
                    if saveMarks is True:
                        f.write('[' + k + ']')
                    f.write(v + '|')
                f.write('\n')
                f.flush()
    
    def parseAmazonDataset(self, inputJSONFile, userDat, itemDat, ratingDat, saveMarks):
        '''
        :param inputJSONFile a file ended with ".json
        
        {
        "reviewerID": "A11N155CW1UV02", 
        "asin": "B000H00VBQ", 
        "reviewerName": "AdrianaM", 
        "helpful": [0, 0], 
        "reviewText": "I had big expectations because I love English TV, in particular Investigative and detective stuff but this guy is really boring. It didn't appeal to me at all.", 
        "overall": 2.0, 
        "summary": "A little bit boring for me", 
        "unixReviewTime": 1399075200, 
        "reviewTime": "05 3, 2014"
        }

        '''
        # create a new directory to store the generated data files
        data_dir = inputJSONFile[:-5]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        user_id, item_id = Str2ID(), Str2ID()
        user_reviews, item_reviews = OrderedDict(), OrderedDict()
        rating_list = []
            
        with open(inputJSONFile, 'r') as f:
            count = 0
            for line in f:
                
                record = json.loads(line)
                
                # assign an integer ID for each user
                userID = user_id.assignID(record["reviewerID"]) + 1
                # assign an integer ID for each item
                itemID = item_id.assignID(record["asin"]) + 1
                # get the ratings of users to items
                rating = record["overall"]
                # get the time-stamps of ratings
                timestamp = record["unixReviewTime"]
                # userID::itemID::rating::timestamp
                rating_list.append(str(userID) + "::" + str(itemID) + "::" + str(rating) + "::" + str(timestamp))
                
                # concatenate all the reviews for each user, and split each review with '|'
                key = str(timestamp) + "," + str(itemID) 
                if userID not in user_reviews: user_reviews[userID] = dict()
                user_reviews[userID][key] = record["reviewText"]
                        
                # concatenate all the reviews for each item, split each review with '|'
                key = str(timestamp) + "," + str(userID) 
                if itemID not in item_reviews: item_reviews[itemID] = dict()
                item_reviews[itemID][key] = record["reviewText"]
                
                # show progress
                count = count + 1
                if count % 10000 == 0: print("Processed %d records..." % (count))
                    
        print("#users:%d\t#items:%d\t#ratings/reviews:%d" % (user_id.size(), item_id.size(), len(rating_list)))
        # save the content of users or items
        print("Save " + data_dir + "\\" + userDat)
        self.writeRawContent(data_dir + "\\" + userDat, user_reviews, saveMarks);
        print("Save " + data_dir + "\\" + itemDat)
        self.writeRawContent(data_dir + "\\" + itemDat, item_reviews, saveMarks);
        # save the data of Str2ID
        print("Save " + data_dir + "\\user.id.map")
        user_id.dump(data_dir + "\\user.id.map");
        print("Save " + data_dir + "\\user.id.map")
        item_id.dump(data_dir + "\\item.id.map");
        # save the ratings data
        print("Save " + data_dir + "\\" + ratingDat)
        with open(data_dir + "\\" + ratingDat, 'w') as f:
            for line in rating_list: f.write(line + '\n')
            
    def parseYelpDataset(self, inputJSONFile, userDat, itemDat, ratingDat, saveMarks):
        '''
        :param inputJSONFile a file ended with ".json, the records are like this:
        
        {
        "review_id":"VfBHSwC5Vz_pbFluy07i9Q",
        "user_id":"cjpdDjZyprfyDG3RlkVG3w",
        "business_id":"uYHaNptLzDLoV_JZ_MuzUA",
        "stars":5,
        "date":"2016-07-12",
        "text":"My girlfriend and I stayed here for 3 nights and loved it...",
        "useful":0,
        "funny":0,
        "cool":0
        }
        '''
        # create a new directory to store the generated data files
        data_dir = inputJSONFile[:-5]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        user_id, item_id = Str2ID(), Str2ID()
        user_reviews, item_reviews = OrderedDict(), OrderedDict()
        rating_list = []
            
        with open(inputJSONFile, 'r') as f:
            count = 0
            for line in f:
                
                record = json.loads(line)
                
                # assign an integer ID for each user
                userID = user_id.assignID(record["user_id"]) + 1
                # assign an integer ID for each item
                itemID = item_id.assignID(record["business_id"]) + 1
                # get the ratings of users to items
                rating = record["stars"]
                # get the time-stamps of ratings
                timestamp = record["date"]
                # userID::itemID::rating::timestamp
                rating_list.append(str(userID) + "::" + str(itemID) + "::" + str(rating) + "::" + str(timestamp))
                
                # concatenate all the reviews for each user, and split each review with '|'
                key = str(timestamp) + "," + str(itemID) 
                if userID not in user_reviews: user_reviews[userID] = dict()
                user_reviews[userID][key] = record["text"]
                        
                # concatenate all the reviews for each item, split each review with '|'
                key = str(timestamp) + "," + str(userID) 
                if itemID not in item_reviews: item_reviews[itemID] = dict()
                item_reviews[itemID][key] = record["text"]
                
                # show progress
                count = count + 1
                if count % 10000 == 0: print("Processed %d records..." % (count))
                    
        print("#users:%d\t#items:%d\t#ratings/reviews:%d" % (user_id.size(), item_id.size(), len(rating_list)))
        # save the content of users or items
        print("Save " + data_dir + "\\" + userDat)
        self.writeRawContent(data_dir + "\\" + userDat, user_reviews, saveMarks);
        print("Save " + data_dir + "\\" + itemDat)
        self.writeRawContent(data_dir + "\\" + itemDat, item_reviews, saveMarks);
        # save the data of Str2ID
        print("Save " + data_dir + "\\user.id.map")
        user_id.dump(data_dir + "\\user.id.map");
        print("Save " + data_dir + "\\user.id.map")
        item_id.dump(data_dir + "\\item.id.map");
        # save the ratings data
        print("Save " + data_dir + "\\" + ratingDat)
        with open(data_dir + "\\" + ratingDat, 'w') as f:
            for line in rating_list: f.write(line + '\n')

        
if __name__ == '__main__':
    reviewProcessor = ReviewProcessor()
    
    reviewProcessor.parseAmazonDataset("data\Amazon_Instant_Video_5.json", "user_content.dat", "item_content.dat", "ratings.dat", False);
    #reviewProcessor.parseAmazonDataset("data\\Apps_for_Android_5.json", "user_content.dat", "item_content.dat", "ratings.dat", False);
    #reviewProcessor.parseAmazonDataset("data\\Kindle_Store_5.json", "user_content.dat", "item_content.dat", "ratings.dat", False);
    
