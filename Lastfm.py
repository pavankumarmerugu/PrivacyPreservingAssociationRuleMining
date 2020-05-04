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
from scipy.interpolate import make_interp_spline, BSpline



data = pd.read_csv('lastfm.csv')

df = data.drop('user', 1)

conv_df = df.astype(bool)

start_time = time.time()

d = apriori(conv_df, min_support=0.01, use_colnames=True, max_len=2)
print((d['itemsets']))


print("--- %s seconds ---" % (time.time() - start_time))

interest_group_centroids = []                               # cluster centriods on which the interest groups are formed
interest_groups = []                                        # Most similar items for each centroid in the interest group
items_len = len(df.columns)                         # lengh of the items in the dataset
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
    centroids = df.columns.values[index]
    interest_group_centroids.append(centroids)
#print('The Centroids are = ', interest_group_centroids, '\n\n')
#For example: The Centroids are =  ['Comedy', 'Electro Hip Hop', 'Jazz Hip Hop', 'Rap/Hip Hop', 'Tropical']


print('\n\n#########################################ITEM-ITEM_SIMILARITY##########################################\n\n')
start_time_sim = time.time()
'# As a first step we normalize the user vectors to unit vectors.'

magnitude = np.sqrt(np.square(df).sum(axis=1))
data_items = df.divide(magnitude, axis='index')

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

print("sim--- %s seconds ---" % (time.time() - start_time_sim))


print('\n\n#########################################INTEREST GROUPS###############################################\n\n')


for i in interest_group_centroids:
    Interest_group = data_matrix.loc[i].nlargest(p).index.values
    print('Interest group', interest_group_centroids.index(i), ' = ', Interest_group, '\n')
    interest_groups.append(Interest_group)
'#print(interest_groups)'

sim_clusuters = len(set(interest_groups[1]).intersection(interest_groups[2])) / len(set(interest_groups[1]).union(interest_groups[2]))
print(sim_clusuters)

print('\n\n#######################FREQUENT-ITEMSETS_APRIORI#######################################################\n\n')

start_time = time.time()
d = apriori(df, min_support=0.1, use_colnames=True, max_len=2)
print((d['itemsets']))

print("--- %s seconds ---" % (time.time() - start_time))

print('#############################################USERS & THEIR LIKES###########################################\n\n')




groups = random.sample(data.user.tolist(),10)
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
        # print(l)
        left.append(l)
        right.append(r)
        # for index in range(1, len(Selected_users_association_IG)):
        # if l in set(Selected_users_association_IG[index]):
        # print(l,'exist')# LHS in user and if LHS present recommend
        if l in set(known_user_likes):
            print('user', user_index, 'gets recommendation:', r)
            R.append(r)
            precision = len(set(known_user_likes).intersection(set(RHS))) / len(set(RHS))
            Recall = len(set(known_user_likes).intersection(set(RHS))) / len(known_user_likes)
            print('precision of user:', user_index, 'is', precision)
            print('Recall of user:', user_index, 'is', Recall)
    #precision_y.append(precision)
    #recall_x.append(Recall)

print(precision_y)
print(recall_x)

"""
fig = plt.figure()
ax = plt.subplot(111)
x = [0.2, 0.4, 0.6, 0.8, 1.0]
y = [1.0, 0.75, 0.5, 0.25]
#Y10 = [1, 0.6, 0.4, 0.3, 0.2]
Y20 = [1.0, 0.5, 0.4, 0.2, 0]
Y30 = [1.0, 0.4, 0.3, 0.1, 0]
Y40 = [1.0, 0.3, 0.2, 0.1, 0]


x_new1 = np.asarray([2, 3, 4, 5])
xnew1 = np.linspace(x_new1.min(), x_new1.max(), 300) #300 represents number of points to make between T.min and T.max
Sim_cluster = [0.37, 0.32, 0.04, 0.09]
spl = make_interp_spline(x_new1, Sim_cluster, k=1)#BSpline object
power_smooth2 = spl(xnew1)
plt.title('Interest group cluster analysis - Lastfm')
plt.xlabel('Interest groups k')
plt.ylabel('Similarity')
ax.plot(xnew1, power_smooth2, 'm')
ax.legend()
plt.show()
"""


x_new = np.asarray([0.2, 0.4, 0.6, 0.8, 1.0])

Y10 = []
Y20 = []
Y30 = []
Y40 = []
"""
fig = plt.figure()
ax = plt.subplot(111)

xnew = np.linspace(x_new.min(), x_new.max(), 300) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(x_new, Y10, k=2)#BSpline object
spl1 = make_interp_spline(x_new, Y20, k=2)#BSpline object
spl2 = make_interp_spline(x_new, Y30, k=2)#BSpline object
spl3 = make_interp_spline(x_new, Y40, k=2)#BSpline object

#spl1 = make_interp_spline(x_new, YANA_doctor, k=2)#BSpline object
power_smooth = spl(xnew)
power_smooth1 = spl1(xnew)
power_smooth2 = spl2(xnew)
power_smooth3 = spl3(xnew)

plt.xlabel('Recall')
plt.ylabel('Precision')
#blue_patch = mpatches.Patch(color='blue', label='Proposed')
#plt.legend(handles=[blue_patch])
#red_patch = mpatches.Patch(color='red', label='YANA')
#plt.legend(handles=[red_patch])
ax.plot(xnew, power_smooth, 'b--', label='K =10')
ax.plot(xnew, power_smooth1, 'm--', label='K=20')
ax.plot(xnew, power_smooth2, 'g--', label='K=30')
ax.plot(xnew, power_smooth3, 'c--', label='K=40')
#ax.plot(xnew, power_smooth1,label='Lastfm')
ax.legend()
plt.title('Lastfm')
plt.show()
"""