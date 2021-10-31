# This file is for to train the baseline dagger algorithm with ray-tune hyper-parameter optimization.
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import notebook
