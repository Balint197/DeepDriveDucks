import math
from gym_duckietown.envs import DuckietownEnv
import argparse

from .teacher import PurePursuitPolicy
from .learner import NeuralNetworkPolicy
from .model import Dronet
from .algorithms import DAgger
from .utils import MemoryMapDataset
import torch
import os
import itertools
import statistics

#new lines
import tensorflow as tf
import datetime
#new lines

# define tensorboard callbacks
# log_dir = "/home/hajna/DeepDriveDucks/iil_baseline/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def launch_env(map_name, randomize_maps_on_reset=False, domain_rand=False):
    environment = DuckietownEnv(
        domain_rand=domain_rand,
        max_steps=math.inf,
        map_name=map_name,
        randomize_maps_on_reset=False,
    )
    return environment


def teacher(env, max_velocity):
    return PurePursuitPolicy(env=env, ref_velocity=max_velocity)


def process_args(each_params, each_save):
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", "-i", default=each_params[0], type=int)
    parser.add_argument("--horizon", "-r", default=each_params[1], type=int)
    parser.add_argument("--learning-rate", "-l", default=each_params[2], type=int)
    parser.add_argument("--decay", "-d", default=each_params[3], type=int)
    parser.add_argument("--save-path", "-s", default=each_save, type=str)
    parser.add_argument("--map-name", "-m", default="loop_empty", type=str)
    parser.add_argument("--num-outputs", "-n", default=2, type=int)
    parser.add_argument("--domain-rand", "-dr", action="store_true")
    parser.add_argument("--randomize-map", "-rm", action="store_true")
    return parser


#semi-automatic gridsearch-based hyperparameter optimalization
#permutations of choosen hyperparameters
#
#opt_params=[episode,
#            horizon.
#            learning_rate,
#            decay,
#            batch_size
#            ]
opt_episode = [15]
opt_horizon = [128, 256]
opt_lr = [1e-1, 1e-2, 1e-3, 1e-4]
opt_decay = [0.5, 0.6, 0.7, 0.8]
opt_batch_size = [32, 64]

opt_params =[opt_episode, opt_horizon, opt_lr, opt_decay, opt_batch_size]
opt_params_permutated = list(itertools.product(*opt_params))
#nr_trials = 2*2*3*4*4 = 192 trials will be

global_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") #defines the hyperopt

mean_losses = []
#naming 
for count, each_params in enumerate(opt_params_permutated):
    
    print("****************************** TRIAL N." + str(count+1) + "/" + str(len(opt_params_permutated))+ " *******************************************")
    each_save = "logs/" + global_tag + "/" + str(each_params) #defines the save path for tensorboard
    
    if __name__ == "__main__":
        parser = process_args(each_params, each_save)
        input_shape = (120, 160)
        batch_size = each_params[4]
        epochs = 15
        #learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        # decays
        #mixing_decays = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        # Max velocity
        max_velocity = 0.5

        config = parser.parse_args()
        # check for  storage path
        if not (os.path.isdir(config.save_path)):
            os.makedirs(config.save_path)
        # launching environment
        environment = launch_env(
            config.map_name,
            domain_rand=config.domain_rand,
            randomize_maps_on_reset=config.randomize_map,
        )

        task_horizon = config.horizon
        task_episode = config.episode

        model = Dronet(num_outputs=config.num_outputs, max_velocity=max_velocity)
        policy_optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate
        )

        dataset = MemoryMapDataset(25000, (3, *input_shape), (2,), config.save_path)
        learner = NeuralNetworkPolicy(
            model=model,
            optimizer=policy_optimizer,
            storage_location=config.save_path,
            batch_size=batch_size,
            epochs=epochs,
            input_shape=input_shape,
            max_velocity=max_velocity,
            dataset=dataset,
        )

        algorithm = DAgger(
            env=environment,
            teacher=teacher(environment, max_velocity),
            learner=learner,
            horizon=task_horizon,
            episodes=task_episode,
            alpha=config.decay,
        )

        #new line
        #new line

        algorithm.train(debug=True) # DEBUG to show simulation
        print("*******EPISODE LAST LOSSES*******")
        print(algorithm.last_losses)  #modified save loss
        print("*******EPISODE MEAN LOSS*******")
        mean = statistics.mean(algorithm.last_losses)
        print(mean)
        mean_losses.append(mean) #save to array
        
        environment.close()
        delete_path = each_save + "/data.dat" #delete temp data.dat file
        os.remove(delete_path)

#print out the best hyperparameters:
min_loss = min(mean_losses)
best_loss_idx = mean_losses.index(min(mean_losses))
hps = opt_params_permutated[best_loss_idx]

print("The best hyperparameters (based on mean loss): ")
hps_text = "episode_number:" + str(hps[0]) + " horizon:" + str(hps[1]) + " learning_rate:" + str(hps[2]) + \
     " decay:" + str(hps[3]) + " batch_size:" + str(hps[4])
print(hps_text)

txt_save_path = "logs/" + global_tag + "/" + "best_hps_" + global_tag +".txt"
f = open(txt_save_path, "w")
f.write(hps_text)
f.write("\nIts average loss: ")
f.write(str(min_loss))
f.close()