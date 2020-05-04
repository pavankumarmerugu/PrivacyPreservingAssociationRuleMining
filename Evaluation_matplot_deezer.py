import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
import random
import subprocess
import sys
from scipy.interpolate import make_interp_spline, BSpline
import seaborn as sns
import matplotlib.pyplot
from sklearn.cluster import KMeans

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
start = time.time()
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

p = (items_len-1) // 5
r = p
length.append(p)

for index in range(0, 4):
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
end_time = time.time()
print("the similarity computation time is--- %s seconds ---" % (end_time - start_time))


#''prints the item-item similarity matrix for all items in DATASET, for example:'

#                   Deezer item-item similarty matrix
#                      Acoustic Blues     ...      West Coast
# Acoustic Blues             1.000000     ...        0.000000
# African Music              0.044191     ...        0.005636
# Alternative                0.008042     ...        0.028171
# Alternative Country        0.037340     ...        0.011230
# Asian Music                0.000000     ...        0.004623


print('\n\n#########################################INTEREST GROUPS###############################################\n\n')


for i in interest_group_centroids:
    Interest_group = data_matrix.loc[i].nlargest(p).index.values
    print('Interest group', interest_group_centroids.index(i), ' = ', Interest_group, '\n')
    interest_groups.append(Interest_group)

sim_clusuters = len(set(interest_groups[1]).intersection(interest_groups[2])) / len(set(interest_groups[1]).union(interest_groups[2]))
print(sim_clusuters)
print('\n\n#######################FREQUENT-ITEMSETS_APRIORI#######################################################\n\n')

start_time = time.time()
d = apriori(df, min_support=0.2, use_colnames=True, max_len=2)
print((d['itemsets']))


print("--- %s seconds ---" % (time.time() - start_time))

print('#############################################USERS & THEIR LIKES###########################################\n\n')

user = [2222]     # The id of the user for whom we want to generate recommendations

user_index = data[data.user == user].index.tolist()[0] # Get the frame index
    #print('user index is: ', user_index)'
known_user_likes = data_items.ix[user_index]
known_user_likes = known_user_likes[known_user_likes > 0].index.values
print('user', user_index, 'likes', known_user_likes, '\n')

groups = random.sample(data.user.tolist(), 20)
print(groups)

user2 = groups     # The id of the user for whom we want to generate recommendations
left = []
right = []
R = []
precision_y = []
recall_x = []

for i in user2:
    user_index = data[data.user == i].index.tolist()[0]  # Get the frame index
    # print('user index is: ', user_index)'
    known_user_likes = data_items.ix[user_index]
    known_user_likes = known_user_likes[known_user_likes > 0].index.values
    print('user', user_index, 'likes', known_user_likes, '\n')


    for i in range(0, len(d['itemsets'])):
        f_s = d['itemsets'][i]
        # print('Recommendation', i, 'is: ', f_s
        LHS = f_s
        RHS = f_s
        l, *_ = LHS
        *_, r = RHS
        #print(r)
        left.append(l)
        right.append(r)
        # for index in range(1, len(Selected_users_association_IG)):
        # if l in set(Selected_users_association_IG[index]):
        # print(l,'exist')# LHS in user and if LHS present recommend
        if l in set(known_user_likes):
            print('user', user_index, 'gets recommendation:', r)
            R.append(r)
            precision = len(set(known_user_likes).intersection(set(R))) / len(set(R))
            Recall = len(set(known_user_likes).intersection(set(R))) / len(known_user_likes)
            #print('precision of user:', user_index, 'is', precision)
            #print('Recall of user:', user_index, 'is', Recall)
    precision_y.append(precision)
    recall_x.append(Recall)

print(precision_y)
print(recall_x)
"""
x = [0.2, 0.4, 0.6, 0.8, 1.0]
y = [1.0, 0.75, 0.5, 0.25]
#Y10 = [1, 0.6, 0.4, 0.3, 0.2]




Y40 = [1.0, 0.3, 0.2, 0.1, 0]

Yana = []
plt.plot(x, Y40)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.yscale('linear')
plt.grid(False)
plt.show()

"""


print('#############################################Accuracy plot###########################################\n\n')
x_new = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])

Y10 = []
Y20 = []
Y30 = []
Y40 = []

""""
fig = plt.figure()
ax = plt.subplot(111)

xnew = np.linspace(x_new.min(), x_new.max(), 300) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(x_new, Y40, k=2)#BSpline object
#spl1 = make_interp_spline(x_new, YANA_doctor, k=3)#BSpline object
power_smooth = spl(xnew)
#power_smooth1 = spl1(xnew)
plt.xlabel('Recall')
plt.ylabel('Precision')
#blue_patch = mpatches.Patch(color='blue', label='Proposed')
#plt.legend(handles=[blue_patch])
#red_patch = mpatches.Patch(color='red', label='YANA')
#plt.legend(handles=[red_patch])
ax.plot(xnew, power_smooth, 'c--', label='K = 40')
#ax.plot(xnew, power_smooth1,label='YANA')
ax.legend()
plt.title('Deezer')
plt.show()

#Similarity = 1 - (len(set(y).intersection(set(Y40))) / len(set(y).union(set(Y40)))) # measures similarity between sets
#print(Similarity)
"""
"""
print('#############################################deezer group plot###########################################\n\n')
fig = plt.figure()
ax = plt.subplot(111)

xnew = np.linspace(x_new.min(), x_new.max(), 300) #300 represents number of points to make between T.min and T.max
#spl = make_interp_spline(x_new, Y10, k=2)#BSpline object
#spl1 = make_interp_spline(x_new, Y20, k=2)#BSpline object
#spl2 = make_interp_spline(x_new, Y30, k=2)#BSpline object
spl3 = make_interp_spline(x_new, Y40, k=2)#BSpline object
#power_smooth = spl(xnew)
#power_smooth1 = spl1(xnew)#
#power_smooth2 = spl2(xnew)
power_smooth3 = spl3(xnew)
plt.xlabel('Recall')
plt.ylabel('Precision')
#blue_patch = mpatches.Patch(color='blue', label='Proposed')
#plt.legend(handles=[blue_patch])
#red_patch = mpatches.Patch(color='red', label='YANA')
#plt.legend(handles=[red_patch])
#ax.plot(xnew, power_smooth, 'b--', label='K=10')
#ax.plot(xnew, power_smooth1, 'm--', label='K=20')
#ax.plot(xnew, power_smooth2, 'g--', label='K=30')
ax.plot(xnew, power_smooth3, 'c--', label='K=40')
ax.legend()
plt.title('Deezer')
plt.show()
"""

"""
print('#############################################Similarity plot###########################################\n\n')

x_new1 = np.asarray([50, 80, 150, 200])
xnew1 = np.linspace(x_new1.min(), x_new1.max(), 300) #300 represents number of points to make between T.min and T.max
Sim_time = [0.2, 0.4, 0.7, 0.95]
spl = make_interp_spline(x_new1, Sim_time, k=3)#BSpline object
power_smooth2 = spl(xnew1)
plt.title('Computation cost of similarity calculation')
plt.xlabel('Items')
plt.ylabel('Time (in seconds)')
plt.plot(xnew1, power_smooth2)
plt.show()
"""

print('#############################################Recommendation plot###########################################\n\n')
"""
x_new1 = np.asarray([50, 80, 150, 200])
xnew1 = np.linspace(x_new1.min(), x_new1.max(), 300) #300 represents number of points to make between T.min and T.max
Sim_time = [0.17, 0.30, 0.53, 0.71]
spl = make_interp_spline(x_new1, Sim_time, k=3)#BSpline object
power_smooth2 = spl(xnew1)
plt.title('Computation cost of recommendation generation')
plt.xlabel('Number of items')
plt.ylabel('Time (in seconds)')
plt.plot(xnew1, power_smooth2)
plt.show()
"""

print('#############################################comparision rec_sim###########################################\n\n')
"""
x_new1 = np.asarray([50, 80, 150, 200])
xnew1 = np.linspace(x_new1.min(), x_new1.max(), 300) #300 represents number of points to make between T.min and T.max
Sim_time = [0.17, 0.30, 0.53, 0.71]

spl = make_interp_spline(x_new1, Sim_time, k=3)#BSpline object
#spl1 = make_interp_spline(x_new, , k=3)#BSpline object

power_smooth2 = spl(xnew1)
plt.title('Computation cost of recommendation generation')
plt.xlabel('Number of items')
plt.ylabel('Time (in seconds)')
plt.plot(xnew1, power_smooth2)
plt.show()


total_time = time.time() - start
print(total_time)
"""
"""
print('#############################################Cluster similarity ###########################################\n\n')


x_new1 = np.asarray([2, 3, 4, 5])
xnew1 = np.linspace(x_new1.min(), x_new1.max(), 300) #300 represents number of points to make between T.min and T.max
Sim_cluster = [0.6, 0.3, 0.29, 0.32]
spl = make_interp_spline(x_new1, Sim_cluster, k=1)#BSpline object
power_smooth2 = spl(xnew1)
plt.title('Interest group cluster analysis - Deezer')
plt.xlabel('Interest groups k')
plt.ylabel('Similarity')
plt.plot(xnew1, power_smooth2)
plt.show()
"""

"""
print('#############################################comparision with yana###########################################\n\n')
fig = plt.figure()
ax = plt.subplot(111)
xnew = np.linspace(x_new.min(), x_new.max(), 300) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(x_new, Y40, k=2)#BSpline object
spl1 = make_interp_spline(x_new, YANA_TvEnt, k=2)#BSpline object

power_smooth = spl(xnew)
power_smooth1 = spl1(xnew)

plt.xlabel('Recall')
plt.ylabel('Precision')
#blue_patch = mpatches.Patch(color='blue', label='Proposed')
#plt.legend(handles=[blue_patch])
#red_patch = mpatches.Patch(color='red', label='YANA')
#plt.legend(handles=[red_patch])
ax.plot(xnew, power_smooth, 'b--', label='Deezer')
ax.plot(xnew, power_smooth1, 'r--', label='Yana')

ax.legend()
plt.title('TvEnt')
plt.show()
"""

