import matplotlib.pyplot as plt
import numpy as np
import datetime 
import os

sign = np.load("./signs/sign.npy")
anomaly_logscore = np.genfromtxt('anom_logscore_teste_1(class)_ver1_.txt', delimiter=',')
#sign[7001500:7001600,1] = [2000 for i in sign[7001500:7001600,1]]


#plt.plot(sign[:,0],sign[:,1])

plt.plot(sign[7000000:7002000,0],sign[7000000:7002000,1])

plt.show()

    