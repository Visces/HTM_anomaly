import matplotlib.pyplot as plt
import numpy as np
import datetime 
import os
from sklearn.preprocessing import StandardScaler
import copy


def open_data(a,b, standardize = False):
    """
    Load the data.
    Arguments:
        a: starting point.
        b: final point.
        standardize: if True, then standardize the data.

    Returns a dict with the following keys:
        sign: the signal
        gabarito: if it's an anomally or not - for comparison purposes
        anom_score: anomaly numpy array
        anom_likelihood: anomaly numpy array 
        anomaly_logscore: anomaly numpy array
    """

    data_ = dict()

    data_['sign'] = np.load("./signs/sign.npy")
    data_['gabarito'] = np.load("./signs/gabarito.npy")

    data_['anomaly_score'] = np.genfromtxt('anomalies/1_anom_score_.txt', delimiter=',')
    data_['anomaly_likelihood'] = np.genfromtxt('anomalies/1_anom_probscore_.txt', delimiter=',')
    data_['anomaly_logscore'] = np.genfromtxt('anomalies/1_anom_logscore_.txt', delimiter=',')

    if standardize == True:
        standardization(data_)

    return data_


def standardization(dict):
    """
    Standardize the data.
    """

    standardize = StandardScaler()
    dict['sign'][:,1:] = standardize.fit_transform(dict['sign'][:,1:])
    print(dict['sign'][:,1:])


def artificial_error(dict_, a , b):
    """
    Inserts an artificial error, must be aplied before the standardization.
    """

    if b-a >= 6000: ## apenas testando a TM ao inserir erros

        a_aux = a + 4500
        b_aux = a + 4550


        dict_['sign'][a_aux:b_aux,1] = [value*3.2 for value in dict_['sign'][a_aux:b_aux,1]]



def plot_(a, b, dados_dict, standardize_ = False, error = False):
    
    """
    Plot the signal, 'gabarito', anomaly score, likelihood and loglikelihood.
    """
    if error == True:
        artificial_error(dados_dict, a, b)
    
    if standardize_ == True:
        standardization(dados_dict)


    fig, axs = plt.subplots(4)
    color = 'tab:orange'
    axs[0].set_ylabel('frequency(Hz)')
    axs[0].set_title('data plot')
    axs[0].plot( dados_dict['sign'][a:b,0], dados_dict['sign'][a:b,1],color = color)

    color ='tab:blue'
    axs[1].set_ylabel('anom_score')
    axs[1].set_title('anomaly score')
    axs[1].plot( dados_dict['sign'][a:b,0], dados_dict['anomaly_score'][:-1],color = color)

    color ='tab:red'
    axs[2].set_ylabel('anom_likelihood')
    axs[2].set_title('anomaly likelihood')
    axs[2].plot( dados_dict['sign'][a:b,0],  dados_dict['anomaly_likelihood'][:-1],color = color)

    color ='tab:brown'
    axs[3].set_ylabel('anom_loglikelihood')
    axs[3].set_xlabel("time(ms)")
    axs[3].set_title('anomaly log likelihood')
    axs[3].plot( dados_dict['sign'][a:b,0],  dados_dict['anomaly_logscore'][:-1],color = color)


    fig.tight_layout()


    fig, axs = plt.subplots(3)
    color = 'tab:orange'
    axs[0].set_ylabel('Hz')
    axs[0].set_title('data plot')
    axs[0].plot( dados_dict['sign'][a:b,0],  dados_dict['sign'][a:b,1],'*',color = color)

    color ='tab:blue'
    axs[1].set_ylabel('boolean anomaly')
    axs[1].set_title('gabarito')
    axs[1].plot( dados_dict['sign'][a:b,0],  dados_dict['gabarito'][a:b],color = color)

    color ='tab:brown'
    axs[2].set_ylabel('anom_log_likelihood')
    axs[2].set_title('anomaly log likelihood')
    axs[2].plot( dados_dict['sign'][a:b,0],  dados_dict['anomaly_logscore'][:-1],color = color)
    fig.tight_layout()
    plt.show()

def plot_whole(dados_dict):
    """
    Plot the whole dataset.
    """
    plt.plot(dados_dict['sign'][:,0], dados_dict['sign'][:,1], color = 'black')
    plt.show()

def plot_aggregate(a,b,dict_):
    """
    Plot the data and the "aggregated" data, i.e plotting 1 every 2 and 1 every 3 records. 
    """


    dados_dict = dict_
    sign_2 = copy.deepcopy(dados_dict['sign'][:,1])
    sign_2 = [sign_2[i] if i%2==0 else float("NaN") for i in range(np.size(sign_2))] ## pick only 1 every 2 inputs. 

    fig, axs = plt.subplots(2)
    axs[0].set_ylabel('x - standard')
    axs[0].set_title('data plot')
    axs[0].plot( dados_dict['sign'][a:b,0],  dados_dict['sign'][a:b,1],'*', color = 'black')

    axs[1].set_ylabel('x - 1 in 2')
    axs[1].set_title('gabarito')
    axs[1].plot( dados_dict['sign'][a:b,0],  sign_2[a:b],'*', color = 'blue')


    sign_3 = copy.deepcopy(dados_dict['sign'][:,1])
    sign_3 = [sign_3[i] if i%3==0 else float("NaN") for i in range(np.size(sign_3))]

    fig, axs = plt.subplots(2)
    axs[0].set_ylabel('x - standard')
    axs[0].set_title('data plot')
    axs[0].plot( dados_dict['sign'][a:b,0],  dados_dict['sign'][a:b,1], color = 'black')

    axs[1].set_ylabel('x - 1 in 3')
    axs[1].set_title('gabarito')
    axs[1].plot( dados_dict['sign'][a:b,0],  sign_3[a:b], color = 'blue')


    plt.show()



def main():

    ############################ PLOT RANGE ##################################

    a=7160000
    b=7180000

    ##########################################################################
    
    data_ = open_data(a,b, True)
    #plot_(a, b, data_,True, False)
    plot_aggregate(a, b, data_)

    ########################## PLOT AGGREGATION ###############################
    #plot_aggregate(0,14000000,data_)



if __name__ == "__main__":
    main()
