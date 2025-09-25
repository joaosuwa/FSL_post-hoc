import os
import pandas as pd
from trainingConfigs.synth import synthTraining
from trainingConfigs.liver_14520_U133A import liver_14520_U133A_Training
from trainingConfigs.xor_multiple import xor_multiple_training
from trainingConfigs.synth_multiple import synth_multiple_training
from trainingConfigs.liver_multiple import liver_multiple_training
from trainingConfigs.spam_multiple import spam_multiple_training
from trainingConfigs.breast_multiple import breast_multiple_training
from trainingConfigs.leukemia_multiple import leukemia_multiple_training
from trainingConfigs.pretrained_multiple import pretrained_multiple_training
from trainingConfigs.tabfn_multiple import tabfn_multiple_training

os.environ["QT_QPA_PLATFORM"] = "offscreen"

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

#spam_multiple_training()
#liver_multiple_training()
#breast_multiple_training()
#leukemia_multiple_training()
#synth_multiple_training()
#xor_multiple_training()
#pretrained_multiple_training(name="titanic-pre-trained", num_of_tests=3, n_epochs_fsl_posthoc=25, learning_rate=0.001, l=0.0025)
#tabfn_multiple_training(name="tabfn")

import torch
import numpy as np
from metrics import Selection_Accuracy

weights = torch.Tensor([0.015474,
0.013646,
0.052291,
0.177393,
0.125302,
0.058281,
0.016333,
0.041147,
0.009032,
0.143605,
0.283448,
0.191669,
0.045119,
0.158752,
0.048427,
0.034436,
0.201658,
0.011812,
0.269146,
0.268231,
0.086477,
0.135672,
0.080337,
0.290024,
0.000358,
0.041559,
0.204959,
0.233645,
0.019708,
0.024880,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000192,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.001509,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000,
0.000000])

print(weights)