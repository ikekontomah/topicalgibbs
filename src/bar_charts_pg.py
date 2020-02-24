import numpy as np
import matplotlib.pyplot as plt
##Log Probability MSE and MSE Train and test using particle gibbs
plt.figure(1)
plt.subplot(211)
N = 4
train_set = (26.451, 7. ,1.7,11.505)
test_set = (29.836,7.4 ,1.8 ,11.036)
trainStd = (1, 1,1,1)
testStd = (1, 1,1,1)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, train_set, width,color='0.5',yerr=trainStd)
p2 = plt.bar(ind, test_set, width,color='tab:orange',bottom=train_set,yerr=testStd)

plt.ylabel('Mean Squared Error')
plt.title('Lowest training and test set MSE(pGibbs)')
plt.xticks(ind, ( 'CTSlices','Ryn','Houses','MSD'))
plt.yticks([5,10,15,20,25,30,35,40,45,50,55,60])
plt.legend((p1[0], p2[0]), ('Training set', 'Test set'))

plt.show()
plt.subplot(212)
train_set_2 = (4.21,1.25,0.54,3.78)
test_set_2 = (4.26,1.27,0.55,3.77)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1_new = plt.bar(ind,train_set_2,width,color='0.5',yerr=trainStd)
p2_new = plt.bar(ind, test_set_2, width,color='tab:orange',bottom=train_set_2,yerr=testStd)

plt.ylabel('-Log Likelihood MSE')
plt.title('Lowest training and test set Loglikelihood MSE(pGibbs)')
plt.xticks(ind, ( 'CTSlices','Ryn','Houses','MSD'))
plt.yticks([0.0,2.0,4.0,6.0,8.0,10.0])
plt.legend((p1_new[0], p2_new[0]), ('Training set', 'Test set'))

plt.show()

#CT-SLICES
#train-mse,log-prob-train,test-mse,log-prob-test
# [ 264.51  , -4.21 , 298.36  , -4.26] #pgibbs


#Ryn

#train-mse,log-prob-train,test-mse,log-prob-test
#[ 0.7  ,-1.25 , 0.74 ,-1.27] p-gibbs

#Houses
#[ 0.17 ,-0.54  ,0.18 ,-0.55] pgibs

#MSD
#[ 115.05  , -3.78 , 110.36 ,  -3.77]