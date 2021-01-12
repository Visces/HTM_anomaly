import matplotlib.pyplot as plt
import numpy as np

import datetime 
import os
#a=7000000,b=7008000
sign = np.load("./signs/sign.npy")
anomaly_logscore = np.genfromtxt('anomalies/3_anom_score_.txt', delimiter=',')

#######################ERRO ARTIFICIAL ################################
sign[7001500:7001600,2] = [1000 for i in sign[7001500:7001600,2]]
#######################################################################

#plt.plot(sign[:,0],sign[:,1])


fig, ax1 = plt.subplots()
color = 'tab:orange'
ax1.set_xlabel('time(s)')
ax1.set_ylabel('frequency(Hz)',color = color)
ax1.plot(sign[7000000:7003000,0],sign[7000000:7003000,2],color = color)
ax1.tick_params(axis='y', labelcolor = color)

ax2 = ax1.twinx()

color ='tab:blue'
ax2.set_ylabel('anom_score')
ax2.plot(sign[7000000:7003000,0], anomaly_logscore[:-1],color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()

    