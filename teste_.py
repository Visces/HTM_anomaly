import nupic
import json 
import csv
import pprint
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

t = getScalarMetricWithTimeOfDayAnomalyParams(  metricData = [0],
                                                minVal = -15.0,
                                                maxVal = 15.0,
                                                tmImplementation = "cpp"    )



pp = pprint.PrettyPrinter(indent = 1)
pp.pprint(t)


