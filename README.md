# DeepD(r)iveDucks

Budapest University of Technology and Economics

Deep learning in practice 2021 (VITMAV45)

Team members: Holló Áron, Tóth Tibor Áron, Hajnal Bálint

---------------------------------------------------------------------
## A feladat kiírás: 

Önvezető autózás a Duckietown környezetben
---------------------------------------------------------------------
Cél: A téma során a csapatnak egy önvezető AI betanítását és
tesztelését kell kiviteleznie a Duckietown szimulációs környezetben. A
tanításhoz szabadon megválasztható a  Deep Learning vagy Deep
Reinforcement Learning algoritmus és a kapcsolódó framework is (pl. Ray
RLlib [1], Dopamine [2], TRFL [3]). Plusz pontot ér ha a sikeresen
betanított ágenst feltöltik a Duckietown hivatalos versenyének
szerverére [4] és/vagy az NVIDIA Jetson Nanoval ellátott Duckiebotra is
és tesztelik a valós környezetben az IB213-as teremben. 

Kiindulás: 

https://docs.duckietown.org/daffy/AIDO/out/index.html [5]

https://github.com/duckietown/gym-duckietown [6]


## Task description: 

Autonomous driving in the Duckietown environment
---------------------------------------------------------------------
Goal: The team has to train and test a self-driving AI in the Duckietown
simulation environment. The Deep Learning or Deep Reinforcement Learning 
algorithm and framework (i.e. RayRLlib [1], Dopamine [2], TRFL [3]) 
can be arbitrarily chosen. If you successfully train the agent and upload
it to the official Duckietown competition server[4] or/and you test it in
real environment on the Duckiebot with NVIDIA Jetson Nano in room IB213
at the University, then you can get additional points.

Starting point:

https://docs.duckietown.org/daffy/AIDO/out/index.html [5]

https://github.com/duckietown/gym-duckietown [6]



Links:
------
[1] https://docs.ray.io/en/latest/rllib.html

[2] https://github.com/google/dopamine

[3] https://github.com/deepmind/trfl

[4] https://challenges.duckietown.org/v4/

[5] https://docs.duckietown.org/daffy/AIDO/out/index.html

[6] https://github.com/duckietown/gym-duckietown


---------------------------------------------------------------------
### INITIALIZE /DOCKER METHOD/:

if you dont have nvidia support, use `docker` instead `nvidia-docker`!

pull our docker image:
`nvidia-docker pull artot/deepdriveducks`

check the pulled image:
`nvidia-docker images`

start a container:
`nvidia-docker run -t -d artot/deepdriveducks`  

check running containers:
`nvidia-docker container ls`

enter into the container + bash terminal:
`nvidia-docker exec -it <your_container_id> bash`

NOW YOU CAN EXECUTE THE TESTS, TRAINS, ETC.

detach without stopping container:
`press Ctrl+P then Ctrl+Q`

stop the running docker container:
`nvidia-docker stop <your_container_id>`

### INITIALIZE /MANUAL METHOD/:

`git clone https://github.com/duckietown/gym-duckietown.git`

`cd gym-duckietown/`

`pip3 install -e .`

`cd ..`

`git clone https://github.com/Balint197/DeepDriveDucks`

`cd DeepDriveDucks`

`pip3 install tensorboard`

`python -m learning.train` for training, and `python -m learning.train_tune` for hyperparameter optimisation


### TRY OUR TRAINING:

test our method with custom hyperparameter optimisation:
`xvfb-run -a -s "-screen 0 1400x900x24" python3 -m learning.train_tune`

test our method with custom hyperparameter optimisation with reduced scearch space:
`xvfb-run -a -s "-screen 0 1400x900x24" python3 -m learning.train_tune_reduced`

test the original dagger training:
`xvfb-run -a -s "-screen 0 1400x900x24" python3 -m learning.train`

---------------------------------------------------------------------


### Baseline algorithm and its authors

Baseline algorithm is from: https://github.com/duckietown/challenge-aido_LF-baseline-dagger-pytorch

```
@phdthesis{diaz2018interactive,
  title={Interactive and Uncertainty-aware Imitation Learning: Theory and Applications},
  author={Diaz Cabrera, Manfred Ramon},
  year={2018},
  school={Concordia University}
}

@inproceedings{ross2011reduction,
  title={A reduction of imitation learning and structured prediction to no-regret online learning},
  author={Ross, St{\'e}phane and Gordon, Geoffrey and Bagnell, Drew},
  booktitle={Proceedings of the fourteenth international conference on artificial intelligence and statistics},
  pages={627--635},
  year={2011}
}

@article{loquercio2018dronet,
  title={Dronet: Learning to fly by driving},
  author={Loquercio, Antonio and Maqueda, Ana I and Del-Blanco, Carlos R and Scaramuzza, Davide},
  journal={IEEE Robotics and Automation Letters},
  volume={3},
  number={2},
  pages={1088--1095},
  year={2018},
  publisher={IEEE}
}
```
