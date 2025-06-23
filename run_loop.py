from active_learning_loop_test import ActiveLearningLoop
import os
import re
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, product
from sklearn.model_selection import KFold, train_test_split
from ase import Atoms, io
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.data import covalent_radii, atomic_numbers
from ase.io import read, write, Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs
from mace.calculators import MACECalculator
from mace.cli.run_train import run
from mace.tools import build_default_arg_parser

# define setup (which elements are in the target structure, how many structures should be generated per complexity etc)
elements = ["C", "O", "Pt"]
r_max = 10.0
cell_size = r_max * 2 + 10.0
total_structures = 3000
dataset_name = "dimers_1000"
nanoparticle_path = "MD_300_43_F_opt_final.xyz"

loop = ActiveLearningLoop(
    elements=elements,
    r_max=r_max,
    cell_size=cell_size,
    total_structures=total_structures,
    dataset_name=dataset_name,
    nanoparticle_path=nanoparticle_path,
    n_points=167,
    max_complexity=8,
    start_from=2,
    top_k = 30,
    mode="start",
    start_model_name="dimers_1000_complexity_2"
)


# === Run the loop ===
loop.run()



# dimer number reduced (I set n_points to 50) 
# log file created
