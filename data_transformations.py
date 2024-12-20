'''data_transformations.py
Maya Kalenak
Performs translation, scaling, and rotation transformations on data
CS 251 / 252: Data Analysis and Visualization
Fall 2024

NOTE: All functions should be implemented from scratch using basic 
NumPy WITHOUT loops and high-level library calls.
'''

import numpy as np

def normalize(data):
    '''Perform min-max normalization of each variable in a dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be normalized.

    Returns:
    -----------
    ndarray. shape=(N, M). The min-max normalized dataset.
    '''
    mins = np.min(data, axis = 0)
    maxs = np.max(data, axis = 0)

    data = (data - mins) / (maxs - mins)

    return data

def center(data):
    '''Center the dataset.

    Parameters:
    -----------
    data: ndarray. shape=(N, M). The dataset to be centered.

    Returns:
    -----------
    ndarray. shape=(N, M). The centered dataset.
    '''
    data = data - data.mean(axis=0)

    return data

def rotation_matrix_3d(degrees, axis='x'):
    '''Make a 3D rotation matrix for rotating the dataset about ONE 
    variable ("axis").

    Parameters:
    -----------
    degrees: float. Angle (in degrees) by which the dataset 
    should be rotated.
    axis: str. Specifies the variable about which the dataset 
    should be rotated. Assumed to be either 'x', 'y', or 'z'.

    Returns:
    -----------
    ndarray. shape=(3, 3). The 3D rotation matrix.

    NOTE: This method just CREATES and RETURNS the rotation matrix. 
    It does NOT actually PERFORM the rotation!
    '''
    degree = np.deg2rad(degrees)
    if axis == 'x':
        return np.array([[1,0,0],[0,np.cos(degree), -np.sin(degree)],
                         [0,np.sin(degree),np.cos(degree)]])
    if axis == 'y':
        return np.array([[np.cos(degree), 0, -np.sin(degree)],
                         [0,1,0], [np.sin(degree),0,np.cos(degree)]])
    if axis == 'z':
        return np.array([[np.cos(degree), -np.sin(degree),0],
                         [np.sin(degree),np.cos(degree),0],[0,0,1]])
    return np.array([[np.cos(degree), -np.sin(degree)],
                     [np.sin(degree),np.cos(degree)]])
