# virtual display setup
!pip install pyvirtualdisplay

from pyvirtualdisplay import Display
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# This code creates a virtual display to draw game images on. 
# If you are running locally, just ignore it
import os
def create_display():
    display = Display(visible=0, size=(1400, 900))
    display.start()
    if type(os.environ.get("DISPLAY")) is not str or len(os.environ.get("DISPLAY"))==0:
        !bash ../xvfb start
        %env DISPLAY=:1

# initialize training
learning_rates = ['1e-1', '1e-2', '1e-3', '1e-4', '1e-5']
mixing_decays = ['0.5', '0.6', '0.7', '0.8', '0.85', '0.9', '0.95']

save_path = "imitation_baseline" #@param {type: "string"}
episode = 10 # @param {type: "integer"}
horizon = 128 # @param {type: "integer"}
learning_rate = "1e-3" # @param ['1e-1', '1e-2', '1e-3', '1e-4', '1e-5']
decay = "0.7" # @param ['0.5', '0.6', '0.7', '0.8', '0.85', '0.9', '0.95']
map_name = "loop_empty" #@param {type: "string"}
# number of outputs can be 2 to predict omega and velocity
# or  1 to fix velocity and predict only omega
num_outputs = 2 # @param {type: "integer"} 
learning_rate = learning_rates.index(learning_rate)
decay = mixing_decays.index(decay)

# start training
create_display()
!python -m learning.train --save-path {save_path} --episode {episode} --horizon {horizon} --learning-rate {learning_rate} --decay {decay} --map-name {map_name} --num-outputs {num_outputs}

#open tensorboard
%load_ext tensorboard
%tensorboard --logdir {save_path}

#test
map_name = "loop_empty" #@param {type: "string"}
episode = 10 # @param {type: "integer"}
horizon = 128 # @param {type: "integer"}

create_display()
!python -m learning.test --model-path {os.path.join(save_path, "model.pt")} --num-outputs {num_outputs} --map-name {map_name} --episode {episode} --horizon {horizon}
