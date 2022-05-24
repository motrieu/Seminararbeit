# Function to get training data
import numpy as np
from get_training_data import linear_isotropic_epsilon_energy_cauchy

n_samples_per_load_case = 30
load_magnitude = 0.0001
mu = 25000 #MPa
lamb = 50000 #MPa

training_data,energy_train, sigma_train,total_number_of_samples = \
    linear_isotropic_epsilon_energy_cauchy(n_samples_per_load_case, load_magnitude,  mu, lamb)
