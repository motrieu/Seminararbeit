import torch
import build_tensor
import analytical_expressions

"""
Function: def linear_isotropic_epsilon_energy_cauchy
application: create data for training or test
computes: infinitesimal strain, energy and cauchy stress
returns: infinitesimal strain, energy, cauchy stress and total number of samples
"""


def linear_isotropic_epsilon_energy_cauchy(n_samples_per_load_case, load_magnitude, mu, lamb):
    number_of_load_cases = 6  # types of loads considered
    total_number_of_samples = n_samples_per_load_case * number_of_load_cases  # total number of samples at the enf of the day
    epsilon_basis = torch.empty(number_of_load_cases, 6)  # load case basis
    epsilon_train = torch.zeros(total_number_of_samples, 6)  # deformation/strain data
    sigma_train = torch.zeros(total_number_of_samples, 6)  # cauchy data
    energy_train = torch.zeros(total_number_of_samples, 1)  # energy data
    delta = torch.linspace(0, load_magnitude, steps=n_samples_per_load_case) # uniform load increment
    counter = 0  # auxiliary variable to get correct index
    for i in range(n_samples_per_load_case): # number of load increments applied to each load case
        epsilon_basis[0][0:] = torch.tensor([delta[i], 0, 0, 0, 0, 0])  # uniaxial stretch
        epsilon_basis[1][0:] = torch.tensor([delta[i], delta[i], 0, 0, 0, 0]) # biaxial stretch
        epsilon_basis[2][0:] = torch.tensor([delta[i], delta[i],delta[i], 0, 0, 0])  # volumetric
        epsilon_basis[3][0:] = torch.tensor([0, 0, 0, delta[i], 0, 0])  # shear in yz
        epsilon_basis[4][0:] = torch.tensor([0, 0, 0, 0, delta[i], 0])  # shear in xz
        epsilon_basis[5][0:] = torch.tensor([0, 0, 0, 0, 0, delta[i]])  # shear in xy
        for j in range(number_of_load_cases):  #  number of load cases
            epsilon_train[counter][:] = epsilon_basis[j][:]   # build strain tensor for input of NN
            local_epsilon_tensor = build_tensor.vector_to_2order_sym_tensor(epsilon_basis[j][:])
            local_energy = analytical_expressions.energy_linear_isotropic(local_epsilon_tensor, mu, lamb)
            local_cauchy_vector = analytical_expressions.stress_cauchy_from_epsilon(local_epsilon_tensor, mu, lamb)
            energy_train[counter][0] = local_energy  # build energy tensor for NN
            sigma_train[counter,:] = local_cauchy_vector[:,0]  # build cauchy tensor for NN
            counter = counter + 1

    return epsilon_train, energy_train, sigma_train, total_number_of_samples

