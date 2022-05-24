import torch
import build_tensor


def moduli_isotropic(mu, lamb):
    moduli = torch.zeros(6, 6)
    moduli[0][0] = lamb + 2 * mu
    moduli[1][1] = moduli[0][0]
    moduli[2][2] = moduli[0][0]
    moduli[0][1] = lamb
    moduli[1][0] = moduli[0][1]
    moduli[0][2] = lamb
    moduli[2][0] = moduli[0][2]
    moduli[1][2] = lamb
    moduli[2][1] = moduli[1][2]
    moduli[3][3] = 2*mu
    moduli[4][4] = 2*mu
    moduli[5][5] = 2*mu
    return moduli


def energy_linear_isotropic(epsilon, mu, lamb):
    elastic_energy = (1 / 2) * lamb * (torch.trace(epsilon)) ** 2 + mu * torch.trace(
            torch.matmul(epsilon, epsilon))
    return elastic_energy


def stress_cauchy_from_epsilon(epsilon, mu, lamb):
    epsilon_vector = build_tensor.epsilon_to_voigt(epsilon)
    moduli = moduli_isotropic(mu, lamb)
    cauchy = torch.matmul(moduli, epsilon_vector)
    return cauchy
