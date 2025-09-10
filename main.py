import os
from trainingConfigs.synth import synthTraining
from trainingConfigs.liver_14520_U133A import liver_14520_U133A_Training
from trainingConfigs.xor_multiple import xor_multiple_training
from trainingConfigs.synth_multiple import synth_multiple_training
from trainingConfigs.liver import liver_multiple_training
from trainingConfigs.tabnet import xor_tabnet_multiple_training

os.environ["QT_QPA_PLATFORM"] = "offscreen"

#liver_multiple_training()
#synth_multiple_training()
xor_multiple_training()
#xor_tabnet_multiple_training()
#xorTraining()
#synthTraining()
#liver_14520_U133A_Training()