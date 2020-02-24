import numpy as np
import matplotlib.pyplot as plt
##Log Probability MSE and MSE Train and test using particle gibbs
plt.figure(1)
plt.subplot(211)
N = 4
train_set = (32.116,7.6 ,2.4,12.543)
test_set = (33.415,8.0 , 2.3 ,11.858)
trainStd = (1, 1,1,1)
testStd = (1, 1,1,1)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, train_set, width,color='0.5',yerr=trainStd)
p2 = plt.bar(ind, test_set, width,color='g',bottom=train_set,yerr=testStd)

plt.ylabel('Mean Squared Error')
plt.title('Lowest training and test set MSE(growprune)')
plt.xticks(ind, ( 'CTSlices','Ryn','Houses','MSD'))
plt.yticks([5,10,15,20,25,30,35,40,45,50,55,60,65])
plt.legend((p1[0], p2[0]), ('Training set', 'Test set'))

plt.show()
plt.subplot(212)
train_set_2 = (4.3,1.28 ,0.71,3.83)
test_set_2 = (4.32,1.31 ,0.7,3.81)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1_new = plt.bar(ind,train_set_2,width,color='0.5',yerr=trainStd)
p2_new = plt.bar(ind, test_set_2, width,color='g',bottom=train_set_2,yerr=testStd)

plt.ylabel('-Log Likelihood MSE')
plt.title('Lowest training and test set Loglikelihood MSE(growprune)')
plt.xticks(ind, ( 'CTSlices','Ryn','Houses','MSD'))
plt.yticks([0.0,2.0,4.0,6.0,8.0,10.0])
plt.legend((p1_new[0], p2_new[0]), ('Training set', 'Test set'))

plt.show()

#CT-SLICES
#train-mse,log-prob-train,test-mse,log-prob-test
# [ 321.16  , -4.3  , 334.15  , -4.32] #growprune

#Ryn
#train-mse,log-prob-train,test-mse,log-prob-test
#[ 0.76 ,-1.28  ,0.8  ,-1.31] $growprune

#Houses
#[ 0.24 ,-0.71 , 0.23 ,-0.7 ]


#MSD
#[ 125.43  , -3.83  ,118.58   ,-3.81]