import nupic
import csv
import matplotlib.pyplot as plt
from datetime import datetime as dt
from nupic.encoders.date import DateEncoder
from nupic.encoders.random_distributed_scalar import RandomDistributedScalarEncoder
from nupic.encoders.date import DateEncoder
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.algorithms.temporal_memory import TemporalMemory
from  nupic.algorithms.anomaly import Anomaly
from nupic.algorithms.anomaly_likelihood import AnomalyLikelihood  
from nupic.frameworks.opf.common_models.cluster_params import getScalarMetricWithTimeOfDayAnomalyParams 
from nupic.frameworks.opf.model_factory import ModelFactory
import numpy as np
import pandas as pd
import os
import time

########### clear function ###################
clear = lambda: os.system('cls')  
clear()

########### data #############################

sign = np.load("./signs/sign.npy")

date = sign[7000000:7005000,0]

scalar = sign[7000000:7005000,1]




scalar = [float(i) for i in scalar]

teste = np.column_stack((date,scalar))

N_DATA, trash = np.shape(teste)

print(N_DATA)


########## finding parameters ################

#didn't undersand very well this parte, so...

#let's try, then. first: https://www.youtube.com/watch?v=-N-2Cv-Q8h8&t=540s at this time...
#it's explained how they got the parameters 

# params = getScalarMetricWithTimeOfDayAnomalyParams(
#     metricData= teste,
#     tmImplementation="tm_cpp",
#     minVal=0.0,
#     maxVal=100.0
# )
# print(params)

########### encoders class ###################
#Generating the encoder classes
scalar_encoder = RandomDistributedScalarEncoder(resolution = 0.03)

#two inputs separated by less than the 'resolution' will have the same encoder output.
time_encoder = RandomDistributedScalarEncoder(resolution = 0.001)

#7 = how much bits represent one input
#0.25 = radius = if an input ir greater than the radius in comparisson with anoter ..
#they won't overlapp 

bits_scalar = np.zeros(scalar_encoder.getWidth())
bits_time = np.zeros(time_encoder.getWidth())


SIZE_ENCODER_ = np.size(bits_time) + np.size(bits_scalar)




############ SPATIAL POOLER #####################

N_COLUMNS = 2048

sp = SpatialPooler(
    inputDimensions = (SIZE_ENCODER_,),
    columnDimensions = ( N_COLUMNS,), # in this case we will use 2048 mini-columns distributed in a "linear array" ...
        
    potentialRadius = SIZE_ENCODER_, # i set the potential radius of each mini-column as the whole ...
    #input space
        
    potentialPct = 0.85, # how many bits on the input space that should have some permanence with each mini-column
    ## attention: having a permanence value doesn't mean that it will be connected, to be connected the "connection...
    ## force" / permanence needs to be higher than the threshold.
        
    globalInhibition = True,  # means that the winning columns are selected in the neighborhood, though in this code...
    #we're dealing as if all the columns are neigbhors with one another
    localAreaDensity = -1.0,
    numActiveColumnsPerInhArea = 40,
    stimulusThreshold = 0,
    ##Well, if we set the number of active columns per input, than there is no need to set an stimulusThreshold
    ##First = because the simulusTHreshold will be already set as the sum of permanences of the 40th column...@@NEED TO CHECK
    ##any other mini-column with less than it won't be active on this input 
        
    synPermInactiveDec = 0.008, #if a column is active, the off (that aren't 1 in input) bits which it is connected ...
    # will have a decrement on the "synapse force"/permance @@
        
    synPermActiveInc = 0.08,#if a column is active, the on (that are 1 in input) bits which it is connected ...
    # will have a increment on the "synapse force"/permance @@
        
    synPermConnected = 0.1, #how much the "strength" of the connection between the on bit and the mini-column ...
    #needs to be for they to be connected. @@

    # @@ what needs to be checked is if the bits with synPermConnected < 0.1 will decrement ou increment when a column ...
    # is active - but i think they will increment

    minPctOverlapDutyCycle = 0.001, #  number between 0 and 1.0, used to set a floor on how often a column 
    #...should have at least stimulusThreshold active inputs  
        
    dutyCyclePeriod = 100, # how many "inputs seen" this should happen
        
    boostStrength = 0.01,

    seed = 47,
    spVerbosity = 0,
    wrapAround = False
    )

################ TEMPORAL MEMORY #################

tm= TemporalMemory(
    columnDimensions= (N_COLUMNS,), # number of columns - it must be the same number of SP's columns

    cellsPerColumn= 15, # num of cells per colums - this parameter dicatates how many different cotexts TM ...
    # could learn, so it's really importante to not select really lower numbers as 2 or 1, and higher numbers ...
    #than 40 can be useless, because the number of cells gets REALLY HIGH (40^2048 in this case)

    activationThreshold= 13, # Segment: usually each cell has only one segment, but some cells could eventually have more...
    #. Segments can be interpreted clusters of distal synapses of one cell in respect to others.  Those segments are created ...
    # during the learning phase, while a column burst and a "winner cell" search for new segments to create context. ...
    # Those segmets are said to be active if the activationThreshold > 13.

    initialPermanence= 0.21, # synapses are "connection" between cells within the layer. The permanence is the "strength" of this...
    # conection. Though, even if 2 cells have a "permance" value for their synapses, it doesn't mean that they will be connected.
    
    connectedPermanence= 0.5, # two cells will only be connected if the synapse permanence is higher than 0.5

    minThreshold= 13, # A segment will only be active/connected if there are more than 10 sucessefull connected synapses within..
    #it.

    maxNewSynapseCount= 20, # How many synapses can be added to a segment during learning
    #incremante and decrement of permanences only occur within a cell activation - same as in SPATIAL POOLER - but here we're talking...
    #about cells

    permanenceIncrement= 0.1, #amount that will be added o a synapse during learning if the synapse is active or potential active
    
    permanenceDecrement= 0.1, #amount that will be decreased if the synapse is in the dendritic distal segment and if the cell ...
    #wasn't active on previous state
    #On each active segment, there will be permanence increase of every active synapse, decrease on every inactive synapse and ...
    #creation of new synapses to cells that were active in previous state

    predictedSegmentDecrement= 0.0004, #punishment for SEGMENTS for incorrect predictions
    #from nupic documentation: predictedSegmentDecrement: A good value is just a bit larger than (the column-level sparsity * permanenceIncrement)...
    #So, if column-level sparsity is 2% and permanenceIncrement is 0.01, this parameter should be something like 4% * 0.01 = 0.0004).

    seed = 1960, 
    maxSegmentPerCell= 255, 
    maxSynapsesPerSegment= 255
)

############## ANOMALY DETECTIONS #############

anom_score = np.zeros((N_DATA+1,))
anom_logscore = np.zeros((N_DATA+1,))

anomaly_score = Anomaly(
        slidingWindowSize = 25)

anomaly_likelihood = AnomalyLikelihood(
        learningPeriod=500,
        historicWindowSize = 213)

dd=0



for i,linha in enumerate(teste):


#####################################################

    scalar_encoder.encodeIntoArray(linha[1], bits_scalar)
    time_encoder.encodeIntoArray(linha[0],bits_time)

    encoder_output = np.concatenate((bits_time, bits_scalar))

    ####################################################

    sdr_output = np.zeros(N_COLUMNS)
    sp.compute(encoder_output,True,sdr_output)
    active_columns = np.nonzero(sdr_output)[0]

    ####################################################

    tm.compute(active_columns, learn=True)

    ####################################################

    anom_score[i] = anomaly_score.compute(tm.getActiveCells(), tm.getPredictiveCells())

    anom_logscore[i] = anomaly_likelihood.computeLogLikelihood(anom_score[i])

    if i%100==0:

        print(i)


anom_score_futuro = np.zeros((5000+1,)) ## excluir isso aqui
anom_logscore_futuro = np.zeros((5000+1,)) ## excluir isso aqui

for i, linha in enumerate(sign[7220000:7225000,:]): ##esse for é o do arquivo teste pra ver se é possível encontrar as anomalias
#excluir esse for depois
    scalar_encoder.encodeIntoArray(linha[1], bits_scalar)
    time_encoder.encodeIntoArray(linha[0],bits_time)

    encoder_output = np.concatenate((bits_time, bits_scalar))

    ####################################################

    sdr_output = np.zeros(N_COLUMNS)
    sp.compute(encoder_output,False,sdr_output)
    active_columns = np.nonzero(sdr_output)[0]

    ####################################################

    tm.compute(active_columns, learn=False)

    ####################################################

    anom_score_futuro[i] = anomaly_score.compute(tm.getActiveCells(), tm.getPredictiveCells())

    anom_logscore_futuro[i] = anomaly_likelihood.computeLogLikelihood(anom_score_futuro[i])

    if i%100==0:

        print(i)

np.savetxt('anom_score_teste_1.txt',anom_score,delimiter=',')

np.savetxt('anom_logscore_teste_1.txt',anom_logscore,delimiter=',')

np.savetxt('anom_score_teste_2.txt', anom_score_futuro,delimiter=',')## excluir isso aqui

np.savetxt('anom_logscore_teste_2.txt',anom_logscore_futuro,delimiter=',')##excluir isso aqui

###################### PLOT ###########################

anom_score = np.genfromtxt('anom_score_teste_1.txt',delimiter=',')
anomaly_logscore = np.genfromtxt('anom_logscore_teste_1.txt', delimiter=',')

x_axis = np.arange(0,N_DATA)

fig, axs = plt.subplots(3)
fig.suptitle('os trem la')
axs[0].plot(date[:],scalar[:])
axs[1].plot(date[:],anom_score[:-1])
axs[2].plot(date[:], anomaly_logscore[:-1])
    

plt.show()






