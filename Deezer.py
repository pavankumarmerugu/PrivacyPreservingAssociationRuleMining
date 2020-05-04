import json
import matplotlib.pyplot as plt
#from pprint import pprint
import csv
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from scipy import sparse
import numpy as np
import time


"""
######################DATASET INFORMATION##########################################
The data was collected from the music streaming service Deezer (November 2017).
These datasets represent friendship networks of users from 3 European countries.
Nodes represent the users and edges are the mutual friendships. We reindexed the
nodes in order to achieve a certain level of anonimity. The csv files contain the
edges -- nodes are indexed from 0. The json files contain the genre preferences of
users -- each key is a user id, the genres loved are given as lists. Genre notations
are consistent across users.In each dataset users could like 84 distinct genres.
Liked genre lists were compiled based on the liked song lists. The countries included
are Romania, Croatia and Hungary. For each dataset we listed the number of nodes an edges.
"""

with open('RO_genres.json') as data_file:
    data = json.load(data_file)

'#print(data.keys())'

users = []                              # Users in the network who uses the service
items = []                              # Items liked by users in the network
recommendations = []                    # Recommendations generated to the users after mining frequent itemsets

for key in data.keys():                 # Retreiving the ID of each user
    users.append(key)

for val in data.values():               # Retrieving the ITEMS liked by each user in the network
    items.append(val)

'#print(users)'
'#Users in the network, for example:'
#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',...,'41772']

'#print(items)'
'#Items liked by all te users in the network, for example:'
#['Dance', 'Soul & Funk', 'Pop', 'Musicals', 'Contemporary R&B', 'Indie Pop', 'Alternative'],

res = items
my_df = pd.DataFrame(res)
my_df.to_csv('out.csv', index=False, header=False)
'#print(my_df.head())'

'# Transposing the items and users into Binary matrix'

te = TransactionEncoder()
te_ary = te.fit(items).transform(items)
df = pd.DataFrame(te_ary, columns=te.columns_)
'#print(df.head())'

'# prints the Binary matrix elements, for example:'
#    Acoustic Blues  African Music     ...      Vocal jazz  West Coast
# 0           False          False     ...           False       False
# 1           False          False     ...           False       False
# 2           False          False     ...           False       False
# 3           False          False     ...           False       False
# 4           False          False     ...           False       False
'#print(te.columns_)'

# Resulting binary matrix to csv file

res = df
my_df = pd.DataFrame(res)
my_df.to_csv('result.csv', index=True, header=True)


data = pd.read_csv('result.csv')
data.rename(columns={'Unnamed: 0': 'user'}, inplace=True)
'#print(data.head())'

'# prints the Binary matrix elements in result.csv, for example:'
# user  Acoustic Blues        ...      Vocal jazz  West Coast
# 0     0           False     ...           False       False
# 1     1           False     ...           False       False
# 2     2           False     ...           False       False
# 3     3           False     ...           False       False
# 4     4           False     ...           False       False


data_items = data.drop('user', 1)

print('Dimension of loaded data is:', np.ndim(data_items))

interest_group_centroids = []                               # cluster centriods on which the interest groups are formed
interest_groups = []                                        # Most similar items for each centroid in the interest group
items_len = len(data_items.columns)                         # lengh of the items in the dataset
length = []  # stores the index of the centroids
print(items_len)
print('\n\n#########################################CENTROIDS#####################################################\n\n')

p = (items_len-1) // 6
r = p
length.append(p)

for index in range(0, 3):
    items_len = int(round(r + p))
    r = items_len
    length.append(items_len)
'#print(length)'
'#Index of the centroid elements, for example:'
#[16, 32, 48, 64, 80]

'# Calculating the centroids based on the length of the items in the DATASET: result.csv'

for index in length:                                        # for each centroid in the length
    centroids = data_items.columns.values[index]
    interest_group_centroids.append(centroids)
#print('The Centroids are = ', interest_group_centroids, '\n\n')
#For example: The Centroids are =  ['Comedy', 'Electro Hip Hop', 'Jazz Hip Hop', 'Rap/Hip Hop', 'Tropical']

print('\n\n#########################################ITEM-ITEM_SIMILARITY##########################################\n\n')
start_time = time.time()
'# As a first step we normalize the user vectors to unit vectors.'

magnitude = np.sqrt(np.square(data_items).sum(axis=1))
data_items = data_items.divide(magnitude, axis='index')

'#print(data_items.head(5))'


def calculate_similarity(data_items):
    data_sparse = sparse.csr_matrix(data_items)
    similarities = cosine_similarity(data_sparse.transpose())
    '#print(similarities)'
    sim = pd.DataFrame(data=similarities, index=data_items.columns, columns=data_items.columns)
    return sim

'# Build the similarity matrix'
data_matrix = calculate_similarity(data_items)
'#print(data_matrix.head())'

#''prints the item-item similarity matrix for all items in DATASET, for example:'
#                      Acoustic Blues     ...      West Coast
# Acoustic Blues             1.000000     ...        0.000000
# African Music              0.044191     ...        0.005636
# Alternative                0.008042     ...        0.028171
# Alternative Country        0.037340     ...        0.011230
# Asian Music                0.000000     ...        0.004623

print("--- %s seconds ---" % (time.time() - start_time))

print('\n\n#########################################INTEREST GROUPS###############################################\n\n')


for i in interest_group_centroids:
    Interest_group = data_matrix.loc[i].nlargest(p).index.values
    print('Interest group', interest_group_centroids.index(i), ' = ', Interest_group, '\n')
    interest_groups.append(Interest_group)
'#print(interest_groups)'

print(set(interest_groups[1]).intersection(interest_groups[3]))

print('\n\n#######################FREQUENT-ITEMSETS_APRIORI#######################################################\n\n')

start_time = time.time()
d = apriori(df, min_support=0.2, use_colnames=True, max_len=5)
print((d['itemsets']))

print("--- %s seconds ---" % (time.time() - start_time))

print('#############################################USERS & THEIR LIKES###########################################\n\n')

user = [2222]     # The id of the user for whom we want to generate recommendations
user_index = data[data.user == user].index.tolist()[0] # Get the frame index
#print('user index is: ', user_index)'
known_user_likes = data_items.ix[user_index]
known_user_likes = known_user_likes[known_user_likes > 0].index.values
print('user', user_index, 'likes', known_user_likes, '\n')


print('#############################################USERS ASSOCIATION TO INTEREST GROUPS##########################\n\n')

Selected_users_association_IG = [user]

for i in interest_groups:
    interest_groups_set = set(i)
    user_likes_set = set(known_user_likes)
    sim_num = user_likes_set.intersection(interest_groups_set)
    sim_den = user_likes_set.union(interest_groups_set)
    sim = len(sim_num)/len(sim_den)

    if sim > 0:
        g = 'User:', user_index, 'is associated to the Interest group:', i, 'with similarity:', sim
        ass_interest_groups = i
        Selected_users_association_IG.append(ass_interest_groups.tolist())

print(Selected_users_association_IG[1])


#user_likes_set.intersection(interest_groups_set)

print('\n\n#########################################CLIENT_SIDE_RECOMMENDATIONS###################################\n\n')

left = []
right = []
R = []

for i in range(0, len(d['itemsets'])):
    f_s = d['itemsets'][i]
    #print('Recommendation', i, 'is: ', f_s
    LHS = f_s
    RHS = f_s
    l, *_ = LHS
    *_, r = RHS
    #print(l)
    left.append(l)
    right.append(r)
#for index in range(1, len(Selected_users_association_IG)):
    #if l in set(Selected_users_association_IG[index]):
         #print(l,'exist')# LHS in user and if LHS present recommend
    if l in set(known_user_likes):
       print('user', user_index, 'gets recommendation:', r)
       R.append(r)



precision = len(set(known_user_likes).intersection(set(R))) / len(set(R))
Recall = len(set(known_user_likes).intersection(set(R))) / len(known_user_likes)


    #print('Items to be checked in users list', l, '\n')
    #print('If item', l, 'is present', 'recommend: ', r, '\n')
