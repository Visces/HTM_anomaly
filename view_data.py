import matplotlib.pyplot as plt
import numpy as np
import datetime 
import os

sign = np.load("./signs/sign.npy")
anomaly_logscore = np.genfromtxt('anom_logscore_teste_2.txt', delimiter=',')



plt.plot(sign[7220000:7225000,0],anomaly_logscore[:-1])
plt.show()

    