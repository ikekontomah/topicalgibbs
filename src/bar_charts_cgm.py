import numpy as np
import matplotlib.pyplot as plt
##Log Probability MSE and MSE Train and test using particle gibbs
plt.figure(1)
plt.subplot(211)
N = 4
train_set = (19.956,  7.5 ,1.9,12.663 )
test_set = (22.084,7.9 , 1.9 ,11.719)
trainStd = (1, 1,1,1)
testStd = (1, 1,1,1)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, train_set, width,color='0.5',yerr=trainStd)
p2 = plt.bar(ind, test_set, width,color='xkcd:sky blue',bottom=train_set,yerr=testStd)

plt.ylabel('Mean Squared Error')
plt.title('Lowest training and test set MSE(CGM)')
plt.xticks(ind, ( 'CTSlices','Ryn','Houses','MSD'))
plt.yticks([5,10,15,20,25,30,35,40,45,50])
plt.legend((p1[0], p2[0]), ('Training set', 'Test set'))

plt.show()
plt.subplot(212)
train_set_2 = (4.05,1.28 ,0.6,3.84)
test_set_2 = (4.09,1.3 ,0.59,3.8)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1_new = plt.bar(ind,train_set_2,width,color='0.5',yerr=trainStd)
p2_new = plt.bar(ind, test_set_2, width,color='xkcd:sky blue',bottom=train_set_2,yerr=testStd)

plt.ylabel('-Log Likelihood MSE')
plt.title('Lowest training and test set Loglikelihood MSE(CGM)')
plt.xticks(ind, ( 'CTSlices','Ryn','Houses','MSD'))
plt.yticks([0.0,2.0,4.0,6.0,8.0,10.0])
plt.legend((p1_new[0], p2_new[0]), ('Training set', 'Test set'))

plt.show()

#CT-SLICES
#train-mse,log-prob-train,test-mse,log-prob-test
# [ 199.56  , -4.05 , 220.84  , -4.09] #cgm

#Ryn
#train-mse,log-prob-train,test-mse,log-prob-test
#[ 0.75 ,-1.28 , 0.79 ,-1.3 ]

#Houses 
#[ 0.19, -0.6  , 0.19 ,-0.59] 

#MSD
#[ 126.63  , -3.84 , 117.19  , -3.8 ]