"""
This module builds tensors according to dimensional requirement. It can be either for regular matrix operation
or to obey a certain notation
"""
import torch


def vector_to_2order_sym_tensor(vector):
    tensor = torch.zeros(3, 3)
    tensor[0][0] = vector[0]
    tensor[0][1] = vector[5]
    tensor[0][2] = vector[4]
    tensor[1][0] = tensor[0][1]  # sym
    tensor[1][1] = vector[1]
    tensor[1][2] = vector[3]
    tensor[2][0] = tensor[0][2]  # sym
    tensor[2][1] = tensor[1][2]  # sym
    tensor[2][2] = vector[2]
    return tensor


def second_to_first_order_tensor(tensor):
    vector = torch.zeros(1, 6)
    vector[0][0] = tensor[0][0]  # xx
    vector[0][1] = tensor[1][1]  # yy
    vector[0][2] = tensor[2][2]  # zz
    vector[0][3] = tensor[1][2]  # yz
    vector[0][4] = tensor[0][2]  # xz
    vector[0][5] = tensor[0][1]  # xy
    return vector


def row_tuple_to_vector(tuple):
    vector = torch.zeros(6)
    a = tuple[0][0]
    vector[0]= tuple[0][0]  # xx
    vector[1] = tuple[0][1]  # yy
    vector[2] = tuple[0][2]  # zz
    vector[3] = tuple[0][3]  # yz
    vector[4] = tuple[0][4]  # xz
    vector[5] = tuple[0][5]  # xy
    return vector


def epsilon_to_voigt(tensor):
    vector = torch.zeros(6, 1)
    vector[0][0] = tensor[0][0]  # xx
    vector[1][0] = tensor[1][1]  # yy
    vector[2][0] = tensor[2][2]  # zz
    vector[3][0] = tensor[1][2]  # yz
    vector[4][0] = tensor[0][2]  # xz
    vector[5][0] = tensor[0][1]  # xy
    return vector


def fourth_to_2order_tangent_moduli_voigt(tensor_fourth):
    tensor_second = torch.zeros(6, 6)
    # first row
    tensor_second[0][0] = tensor_fourth[0][0][0][0]
    tensor_second[0][1] = tensor_fourth[0][0][1][1]
    tensor_second[0][2] = tensor_fourth[0][0][2][2]
    tensor_second[0][3] = tensor_fourth[0][0][1][2]
    tensor_second[0][4] = tensor_fourth[0][0][0][2]
    tensor_second[0][5] = tensor_fourth[0][0][0][1]
    # second row
    tensor_second[1][0] = tensor_second[0][1]  # sym
    tensor_second[1][1] = tensor_fourth[1][1][1][1]
    tensor_second[1][2] = tensor_fourth[1][1][2][2]
    tensor_second[1][3] = tensor_fourth[1][1][1][2]
    tensor_second[1][4] = tensor_fourth[1][1][0][2]
    tensor_second[1][5] = tensor_fourth[1][1][0][1]
    # third row
    tensor_second[2][0] = tensor_second[0][2]  # sym
    tensor_second[2][1] = tensor_second[1][2]  # sym
    tensor_second[2][2] = tensor_fourth[2][2][2][2]
    tensor_second[2][3] = tensor_fourth[2][2][1][2]
    tensor_second[2][4] = tensor_fourth[2][2][0][2]
    tensor_second[2][5] = tensor_fourth[2][2][0][1]
    # fourth row
    tensor_second[3][0] = tensor_second[0][3]  # sym
    tensor_second[3][1] = tensor_second[1][3]  # sym
    tensor_second[3][2] = tensor_second[2][3]  # sym
    tensor_second[3][3] = tensor_fourth[2][1][1][2]
    tensor_second[3][4] = tensor_fourth[1][2][0][2]
    tensor_second[3][5] = tensor_fourth[1][2][0][1]
    # fifth row
    tensor_second[4][0] = tensor_second[0][4]  # sym
    tensor_second[4][1] = tensor_second[1][4]  # sym
    tensor_second[4][2] = tensor_second[2][4]  # sym
    tensor_second[4][3] = tensor_second[3][4]  # sym
    tensor_second[4][4] = tensor_fourth[2][0][0][2]
    tensor_second[4][5] = tensor_fourth[0][2][0][1]
    # sixth row
    tensor_second[5][0] = tensor_second[0][5]  # sym
    tensor_second[5][1] = tensor_second[1][5]  # sym
    tensor_second[5][2] = tensor_second[2][5]  # sym
    tensor_second[5][3] = tensor_second[3][5]  # sym
    tensor_second[5][4] = tensor_second[4][5]  # sym
    tensor_second[5][5] = tensor_fourth[1][0][0][1]
    return tensor_second


def cauchy_from_voigt_to_tensor(vector):
    tensor = torch.zeros(3,3)
    tensor[0][0] = vector[0]  # xx
    tensor[1][1] = vector[1] # yy
    tensor[2][2] = vector[2]  # zz
    tensor[1][2] = vector[3] # yz
    tensor[0][2] = vector[4]  # xz
    tensor[0][1] = vector[5]  # xy
    tensor[2][1] = tensor[1][2]
    tensor[2][0] = tensor[0][2]
    tensor[1][0] = tensor[0][1]
    return tensor


def from_6by6_moduli_to_21_input_row_vector(moduli):
    moduli_vector = torch.zeros(1,21)
    # from first row of moduli
    moduli_vector[0][0] = moduli[0][0]
    moduli_vector[0][1] = moduli[0][1]
    moduli_vector[0][2] = moduli[0][2]
    moduli_vector[0][3] = moduli[0][3]
    moduli_vector[0][4] = moduli[0][4]
    moduli_vector[0][5] = moduli[0][5]
    # from second row of moduli
    moduli_vector[0][6] = moduli[1][1]
    moduli_vector[0][7] = moduli[1][2]
    moduli_vector[0][8] = moduli[1][3]
    moduli_vector[0][9] = moduli[1][4]
    moduli_vector[0][10] = moduli[1][5]
    # from third row of moduli
    moduli_vector[0][11] = moduli[2][2]
    moduli_vector[0][12] = moduli[2][3]
    moduli_vector[0][13] = moduli[2][4]
    moduli_vector[0][14] = moduli[2][5]
    # from fourth row of moduli
    moduli_vector[0][15] = moduli[3][3]
    moduli_vector[0][16] = moduli[3][4]
    moduli_vector[0][17] = moduli[3][5]
    # from fifth row of moduli
    moduli_vector[0][18] = moduli[4][4]
    moduli_vector[0][19] = moduli[4][5]
    #from sixth row of moduli
    moduli_vector[0][20] = moduli[5][5]
    return moduli_vector

def from_21_input_vector_to_tangent_moduli(vector):
    moduli = torch.zeros(6,6)
    counter = 0
    k = 0
    for i in range(6):
        for j in range(k,6):
            moduli[i][j] = vector[counter]
            counter = counter + 1
        for l in range(0,k):
            moduli[i][l] = moduli[l][i]
        k = k+1
    return moduli


def from_21input_vector_to_lower_trig(vector):
    lower_trig = torch.zeros(6,6)
    counter = 0
    k=0
    for j in range(6):
        for i in range(k,6):
            lower_trig[i][j] = vector[counter]
            counter =  counter +1
        k = k+1
    return lower_trig

def from_lower_trig_to_21_input_vector(lower_trig):
    vector = torch.zeros(21)
    k = 0
    counter = 0
    for j in range(6):
        for i in range(k,6):
            vector[counter] = lower_trig[i][j]
            counter = counter+1
        k = k+1
    return vector
