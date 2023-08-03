import numpy as np
from numba import jit
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize



#initialization
L =60
N = 100
Field_of_perception = 270
interior_angle = 0.5*np.radians(Field_of_perception)
eta = 0.05
dt = 0.1
time_step =5000
v = 3
turning_rate = np.radians(40)
rr = 1
dro=12
dra=14
ro = rr+dro
ra = ro+dra
filename='simulation_data'
np.savez(filename)
runtime=100
comr = 6

# MIC periodic distance
@jit(nopython=True)
def periodic_distance(point1, point2, box_size):
    diff = point1 - point2
    diff = np.mod(diff + box_size / 2, box_size) - box_size / 2
    return np.sqrt(np.sum(diff ** 2))

def angle_between(point1, point2, dir1):
    v1 = dir1
    v2 = (point2 - point1)
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    zero_mask1 = (norm1 == 0)
    zero_mask2=(norm2 == 0)
    norm1[zero_mask1] = 1
    norm2[zero_mask2] = 1
    cos_angles = np.sum(v1 * v2, axis=-1) / (norm1 * norm2)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    return np.arccos(cos_angles)


def periodic(pos_array,L):
    #signs!!!!!!
    diff=pos_array-pos_array[:,np.newaxis,:]
    uper=(diff>L/2)
    lower=(diff<-L/2)
    return np.where(uper, -L,0)+np.where(lower,L,0)

def angle_between_p(point1, point2, dir1,pos_array,L):
    dif_b=periodic(pos_array,L) ###############
    v1 = dir1
    v2 = (point2 - point1) +dif_b
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    zero_mask1 = (norm1 == 0)
    zero_mask2 = (norm2 == 0)
    norm1[zero_mask1] = 1
    norm2[zero_mask2] = 1
    cos_angles = np.sum(v1 * v2, axis=-1) / (norm1 * norm2)
    cos_angles = np.clip(cos_angles, -1.0, 1.0)
    return np.arccos(cos_angles)

def field_of_perception(pos_array, dir_array, interior_angle,L):
    angles = angle_between_p(pos_array[:, None], pos_array[None, :],np.reshape(np.tile(dir_array[:, None], (1, 1, N)), (N, N, 3)),pos_array,L )
    np.fill_diagonal(angles,0)
    mask = (angles <= interior_angle)
    mask[np.isnan(angles)] = False
    return mask

def tendency(pos_matrix, pos_array,mask,L):
    dif_b = periodic(pos_array,L)###############
    dif = (pos_matrix - pos_array.reshape(N, 1, 3))+dif_b
    dif*=mask[:, :, np.newaxis]

    squared_dir_matrix = dif ** 2
    squared_sum = squared_dir_matrix.sum(axis=-1)
    norm_v = np.zeros_like(dif)

    nonzero_mask = squared_sum != 0
    norm_v[nonzero_mask] = dif[nonzero_mask] / np.sqrt(squared_sum[nonzero_mask])[:, np.newaxis]

    norm_v[np.isnan(norm_v)] = 0
    norm_v *= mask[:, :, np.newaxis]
    dr = np.sum(norm_v, axis=1)
    return normalize(dr, axis=1, norm='l2')


def align(dir_matrix,mask,pos_array,L):
    squared_dir_matrix = dir_matrix ** 2
    squared_sum = squared_dir_matrix.sum(axis=-1)
    norm_v = np.zeros_like(dir_matrix)

    nonzero_mask = squared_sum != 0
    norm_v[nonzero_mask] = dir_matrix[nonzero_mask] / np.sqrt(squared_sum[nonzero_mask])[:, np.newaxis]

    norm_v[np.isnan(norm_v)] = 0
    norm_v*=mask[:, :, np.newaxis]
    dr=np.sum(norm_v, axis=1)
    return normalize(dr, axis=1, norm='l2')

def neighbor(pos_array, radius_down, radius_up):
    dist = cdist(pos_array, pos_array, metric=lambda x, y: periodic_distance(x, y, L))
    neighbors = np.logical_and(radius_down < dist, dist < radius_up)
    return neighbors

def rotate_vector(v1, v2, theta):

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = theta
    rot_matrix = np.array([[np.cos(angle) + axis[0] ** 2 * (1 - np.cos(angle)),
                            axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
                            axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
                           [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
                            np.cos(angle) + axis[1] ** 2 * (1 - np.cos(angle)),
                            axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
                           [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
                            axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
                            np.cos(angle) + axis[2] ** 2 * (1 - np.cos(angle))]])
    v = np.squeeze(normalize([rot_matrix.dot(v1)],'l2'))

    return v


def turning(new_dir_array, dir_array, turning_rate):
    angles=angle_between(np.zeros((N,3)), dir_array, new_dir_array)
    mask=angles<turning_rate * dt
    indexs=np.squeeze(np.argwhere(mask == False))
    for i in indexs:
        new_dir_array[i] = rotate_vector(dir_array[i], new_dir_array[i], turning_rate * dt)
    # for i in range(len(new_dir_array)):
    #     a = angle(new_dir_array[i], dir_array[i])
    #     if np.abs(a) > turning_rate * dt:
    #         new_dir_array[i] = rotate_vector(dir_array[i],new_dir_array[i],  turning_rate * dt)
    # for i in range(len(new_dir_array)):
    #     a = angle(new_dir_array[i], dir_array[i])
    #     if np.abs(a) > turning_rate * dt:
    #         new_dir_array[i] = rotate_vector(dir_array[i],new_dir_array[i],  turning_rate * dt)

    return new_dir_array
def generate_random_neighbors(neighbors):
    return np.random.randint(neighbors)

def random_selected_partical(matrix,indices):
    """

    :param matrix: nighbours matrix(boolean matrix)
    :param indices: 1d array that contain nth True value(the random select partical)
    :return: the index of each selected particals
    """
    true_palce = np.column_stack(np.where(matrix))
    split_arr = np.split(true_palce[:, 1], np.unique(true_palce[:, 0], return_index=True)[1][1:])
    result = np.empty(len(split_arr))
    for i in range(len(split_arr)):
        if np.isnan(indices[i]):
            result[i] = i
        else:
            result[i] = split_arr[i][indices[i]]
    return result.astype(int)

def update(pos_array, dir_array, rr, ro, ra, turning_rate,L,comr):
    filed = field_of_perception(pos_array, dir_array, interior_angle,L)
    angles=angle_between(np.zeros((N,3)),dir_array[ None,:],dir_array[:, None])
    neighr, neigho, neigha, neigh = neighbor(pos_array, 0, rr), \
                                    neighbor(pos_array, rr, ro ) * filed, \
                                    neighbor(pos_array,ro ,ra)* filed, \
                                    neighbor(pos_array, 0, ra)* filed

    in_r = (np.sum(neighr, axis=1) > 0)
    in_o = (np.sum(neigho, axis=1) > 0)
    in_a = (np.sum(neigha, axis=1) > 0)
    around = (np.sum(neigh, axis=1) > 0)

    pos_matrix = np.tile(pos_array, (N, 1)).reshape(N, N, 3)
    dir_matix=np.tile(dir_array, (N, 1)).reshape(N, N, 3)

    dr = -1*tendency(pos_matrix , pos_array,neighr,L)
    do = align(dir_matix,neigho,pos_array,L)
    da = tendency(pos_matrix , pos_array,neigha,L)

    in_o_not_r = in_o * ~in_r
    in_a_not_r = in_a * ~in_r

    in_o_and_a = in_o_not_r * in_a_not_r

    just_in_o = in_o_not_r * ~in_o_and_a
    just_in_a = in_a_not_r * ~in_o_and_a

    ##############
    # ne=neigh

    ne = neighbor(pos_array, 0, comr)
    np.fill_diagonal(ne, True)
    neigh_dir=dir_matix * ne[:, :, np.newaxis]
    num_neighbors=np.sum(ne, axis=1)

    r_l = np.linalg.norm(np.sum(neigh_dir, axis=1),axis=1) / (num_neighbors)
    # communication cost
    c_l = s * comr / L  # 1-d array
    # payoff earned by an agent l
    p_l = r_l - alp * c_l  # 1-d array
    p_l[np.where(num_neighbors == 1)]=0
    # Fermi rule
    index_of_randomly_partical = random_selected_partical(ne, generate_random_neighbors(num_neighbors))
    p_m = p_l[index_of_randomly_partical]
    w = 1 / (1 + (np.exp((p_l - p_m) / beta)))
    random_selected_particla_s = s[index_of_randomly_partical]
    convert = random_selected_particla_s - s
    w[np.where(convert == -1)] = 1 - w[np.where(convert == -1)]
    w[np.where(convert == 0)] = s[np.where(convert == 0)]
    new_s = []
    for i in range(len(s)):
        new_s.append(np.random.choice([0, 1], p=[1 - w[i], w[i]]))

    new_s = np.array(new_s)
    #############

    new_dir = dr * in_r[:, np.newaxis] + do * just_in_o[:, np.newaxis] + da * just_in_a[:, np.newaxis] + 0.5 * (
            do + da) * in_o_and_a[:, np.newaxis] + dir_array * ~around[:, np.newaxis]
    new_dir=normalize(new_dir,axis=1)
    err = np.random.normal(loc=0, scale=eta, size=(N, 3))
    new_dir += err
    random_theta = normalize(np.random.uniform(low=-1, high=1, size=(N, 3)), axis=1, norm='l2')
    new_dir[np.where(new_s == 0)] = random_theta[np.where(new_s == 0)] #random but turning angle
    update_dir = turning(new_dir, dir_array, turning_rate)
    # update_dir[np.where(new_s == 0)] = random_theta[np.where(new_s == 0)]
    return normalize(update_dir,axis=1,norm='l2'), new_s,0.5*np.sum(angles)/(N**2-N),r_l,p_l



for i in range(runtime):
    array_name=f'arr_{i}'
    an=[]
    for alp in np.arange(0.05,1.05,0.05):
        for beta in np.arange(0.05,1.05,0.05):
            pos_array = L * np.random.rand(N, 3)
            dir_array = normalize(np.random.uniform(low=-1, high=1, size=(N, 3)), axis=1, norm='l2')
            s = np.random.randint(2, size=N)
            for i in range(time_step):
                dir_array,st,angles,r_l,p_l= update(pos_array, dir_array, rr, ro, ra, turning_rate,L,comr)
                vel_array = v * dir_array
                pos_array += vel_array * dt
                pos_array = np.mod(pos_array, L)
                s = st
            an.append(angles)
    data = np.load('simulation_data.npz')
    data = dict(data)
    data[array_name]=an
    np.savez(filename,**data)
    print(array_name)
print('done!')