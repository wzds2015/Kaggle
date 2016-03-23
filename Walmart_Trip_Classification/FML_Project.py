#! /usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from robsvm import robsvm 
from cvxopt import matrix
import svmcmpl
from sklearn import cross_validation
from sklearn import svm

train_file = 'train.csv'
test_file = 'test.csv'

train = pd.read_csv('train.csv')
label = train[['TripType']].values
submit_feature = pd.read_csv('test.csv')

train.drop(['TripType','Upc','FinelineNumber'], axis=1, inplace=True)
submit_feature.drop(['Upc', 'FinelineNumber'], axis=1, inplace=True)
xtrain, xlabel = np.array(train), np.array(label)
tfeature = np.array(submit_feature) 

line, col = xtrain.shape
new_data = []
previous_visit = 0
t_dict = {}
dp_N = 0
vs_N = 0
dp_t_dict = {}
vs_dict = {}
weekday_dict = {'Monday':np.array([1,0,0,0,0,0,0]).reshape(1,7), 'Tuesday':np.array([0,1,0,0,0,0,0]).reshape(1,7), 'Wednesday':np.array([0,0,1,0,0,0,0]).reshape(1,7), 'Thursday':np.array([0,0,0,1,0,0,0]).reshape(1,7), 'Friday':np.array([0,0,0,0,1,0,0]).reshape(1,7), 'Saturday':np.array([0,0,0,0,0,1,0]).reshape(1,7), 'Sunday':np.array([0,0,0,0,0,0,1]).reshape(1,7)} 

for ni in xrange(line):
  if (type(xtrain[ni,3]) is not str):
    continue
  if xtrain[ni,0] != previous_visit:
    vs_N = vs_N + 1
    vs_dict[xtrain[ni,0]] = vs_N - 1  
    previous_visit = xtrain[ni,0]
  if ((xtrain[ni,3] not in dp_t_dict)):
    dp_t_dict[xtrain[ni,3]] = dp_N
    dp_N = dp_N + 1

print "total number of department is: ", dp_N
print "total visit is: ", vs_N
print "All types are: ", dp_t_dict

train_mat = np.zeros((vs_N,dp_N+7), dtype=np.float32)
previous_visit = 0
label_list = []
all_label = []
for ni in xrange(line):
  if (type(xtrain[ni,3]) is not str):
    continue
  if xtrain[ni,0] == previous_visit:
    train_mat[vs_dict.get(xtrain[ni,0]),dp_t_dict.get(xtrain[ni,3])] = xtrain[ni,2]
  else:
    train_mat[vs_dict.get(xtrain[ni,0]),:7] = weekday_dict.get(xtrain[ni,1])
    label_list.append(xlabel[ni])
    if xlabel[ni] not in all_label:
      all_label.append(xlabel[ni])
    previous_visit = xtrain[ni,0]

label_array = np.array(label_list).astype(np.float32).reshape(len(label_list),1)
all_label = np.array([item for sublist in all_label for item in sublist])
label_list = [item for sublist in label_list for item in sublist]
#list(label_array.flatten()).count(999)
#exit(0)
dict_c = dict((i, label_list.count(i)) for i in all_label)

#plt.bar(range(len(dict_c)), dict_c.values(), align='center')
#plt.xticks(range(len(dict_c)), dict_c.keys())
#plt.show()

print "Total Visit is: ", train_mat.shape
print "Total Label is: ", label_array.shape
print "All trip types are: ", all_label   
print "Number of trip type: ", all_label.size
print "All counts are: ", dict_c

#### Test classes ####
#class_1 = 5 
#class_2 = 41
class_1 = 8
class_2 = 9

test_line = dict_c.get(class_1) + dict_c.get(class_2)
test_col = dp_N+7
test_train = np.zeros((test_line,test_col))
test_label = np.zeros(test_line)
k = 0
p = 0
for ni in xrange(label_array.size):
  if label_array[ni] == class_1:
    test_label[k] = -1
    test_train[k,:] = train_mat[ni,:]
    k = k + 1
  if label_array[ni] == class_2:
    test_label[k] = 1
    test_train[k,:] = train_mat[ni,:]
    k = k + 1   ### Total data 
    p = p + 1   ### Data with positive labels

print "Number of test data are: ", test_train.shape  

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions

'''
#fig, ax = plt.subplots()


fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=20, azim=30)
X_reduced = PCA(n_components=3).fit_transform(test_train)
#X_reduced = PCA(n_components=3).fit_transform(test_train)

B = np.random.randint(test_label.size,size=8000)
X_reduced_plot = X_reduced[B,:]
test_label_plot = test_label[B]

ax.scatter(X_reduced_plot[:, 0], X_reduced_plot[:, 1], X_reduced_plot[:, 2], c=test_label_plot,cmap=plt.cm.Paired, alpha=0.75)
#ax.scatter(X_reduced_plot[:, 0], X_reduced_plot[:, 1], c=test_label_plot, cmap=plt.cm.Paired, alpha=1)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])
plt.show()
'''

#### RO Modeling
#### Matrix preparing
### Cross Validation for tunning parameters
c_vec = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 10000, 100000, 1000000]
c_error_ro = np.zeros(len(c_vec))
c_error_svm = np.zeros(len(c_vec))
for ni in xrange(len(c_vec)):

  rs = cross_validation.ShuffleSplit(test_label.size, n_iter=5, test_size=0.2, random_state=0)
  k = 0
  accum_accuracy_RO = 0.
  accum_accuracy_SVM = 0.
  for train_index, test_index in rs:
    temp_train_feature = test_train[train_index,:]
    temp_test_feature = test_train[test_index,:]
    temp_train_label = test_label[train_index]
    temp_test_label = test_label[test_index]
    test_train_pos = temp_train_feature[temp_train_label==1,:]
    test_label_pos = temp_train_label[temp_train_label==1]
    test_train_neg = temp_train_feature[temp_train_label==-1,:]
    test_label_neg = temp_train_label[temp_train_label==-1]
    mu_pos = np.mean(test_train_pos, axis=0)
    mu_neg = np.mean(test_train_neg, axis=0)
    cov_pos = np.cov(test_train_pos.T)
    cov_neg = np.cov(test_train_neg.T)
    X_mat = matrix(temp_train_feature)
    d_vec = matrix(temp_train_label)
    e_vec = matrix((np.abs(temp_train_label+1)<0.00001).astype(np.int).reshape(temp_train_label.size,1))
    P_list = [matrix(cov_pos), matrix(cov_neg)]
    C = c_vec[ni]
    gamma = 1.0
    kernel = 'linear'
    sigma = 1.0

    w,b,u,v,iterations = robsvm(X_mat,d_vec,C,P_list,e_vec)
  
    ### SKL SVC
    clf = svm.SVC(gamma=0.001, C=C)
    clf.fit(temp_train_feature,temp_train_label)

    ### This is too slow
    #sol = svmcmpl.softmargin(X_mat,d_vec,gamma,kernel,sigma)
    #print "Linear SVM solution is: ", sol
    ###

    #### Validation
    predict_label = np.sign(np.dot(temp_test_feature,w)+b)
    score = predict_label * temp_test_label.reshape(predict_label.size,1)
    score[score<0] = 0
    acc_rate = np.sum(score) / float(temp_test_label.size)
    print "Accuracy of (RO) round ",k+1, " is: ", acc_rate, ", C = ", C
    accum_accuracy_RO += acc_rate
  
    predict_label = clf.predict(temp_test_feature) 
    score = predict_label.reshape(predict_label.size,1) * temp_test_label.reshape(predict_label.size,1)
    print score.shape
    score[score<0] = 0
    acc_rate = np.sum(score) / float(temp_test_label.size)
    print "Accuracy of (SVM) round ",k+1, " is: ", acc_rate, ", C = ", C
    accum_accuracy_SVM += acc_rate
  c_error_ro[ni] = accum_accuracy_RO/5
  c_error_svm[ni] = accum_accuracy_SVM/5
  print "Accumulative accuracy of RO is: ", accum_accuracy_RO/5, ", C = ", C
  print "Accumulative accuracy of SVM is: ", accum_accuracy_SVM/5

print "Best C for RO is: ", c_vec[np.argmax(c_error_ro)], ", Accumulative accuracy is: ", c_error_ro.max()
print "Best C for SVM is: ", c_vec[np.argmax(c_error_svm)], ", Accumulative accuracy is: ", c_error_svm.max() 

fig, ax = plt.subplots()
ax.plot(np.log10(c_vec), 1-c_error_svm, color='#04B431', linewidth=2)
ax.plot(np.log10(c_vec), 1-c_error_ro, color='#1138C4', linewidth=2)
ax.set_xlim([-2.2, 6.2])
ax.set_ylim([0.29, 0.34])
temp_name = "./CV_error" + ".png"
plt.savefig(temp_name)
plt.close()

