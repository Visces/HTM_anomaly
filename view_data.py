import matplotlib.pyplot as plt
import numpy as np
import datetime 
import os

#########################CARREGANDO OS DADOS##############################

sign = np.load("./signs/sign.npy")


anomaly_score = np.genfromtxt('anomalies/1_anom_score_.txt', delimiter=',')
anomaly_likelihood = np.genfromtxt('anomalies/1_anom_probscore_.txt', delimiter=',')
anomaly_logscore = np.genfromtxt('anomalies/1_anom_logscore_.txt', delimiter=',')

##########################################################################

####################### INTERVALO ANALISADO ##############################

a = 7000000
b = 7003000

##########################################################################

#######################ERRO ARTIFICIAL ###################################

sign[7001500:7001550,1] = [i+10 for i in sign[7001500:7001550,1]]

##########################################################################

############################ PLOT ########################################

fig, axs = plt.subplots(4)
color = 'tab:orange'
axs[0].set_ylabel('frequency(Hz)')
axs[0].set_title('data plot')
axs[0].plot(sign[a:b,0],sign[a:b,1],color = color)

color ='tab:blue'
axs[1].set_ylabel('anom_score')
axs[1].set_title('anomaly score')
axs[1].plot(sign[a:b,0], anomaly_score[:-1],color = color)

color ='tab:red'
axs[2].set_ylabel('anom_likelihood')
axs[2].set_title('anomaly likelihood')
axs[2].plot(sign[a:b,0], anomaly_likelihood[:-1],color = color)

color ='tab:brown'
axs[3].set_ylabel('anom_loglikelihood')
axs[3].set_xlabel("time(ms)")
axs[3].set_title('anomaly log likelihood')
axs[3].plot(sign[a:b,0], anomaly_logscore[:-1],color = color)


fig.tight_layout()
plt.show()
