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
#from trainingConfigs.pretrained_multiple import pretrained_multiple_training
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
xor_multiple_training()
#pretrained_multiple_training(name="titanic-pre-trained", num_of_tests=3, n_epochs_fsl_posthoc=25, learning_rate=0.001, l=0.0025)
#tabfn_multiple_training(name="tabfn")