import numpy as np
import copy

from lib.transform import compute_transformation


def rotate_element(element, angle, centre_of_rotation):

    # Calculate transformation matrix
    transformation_matrix = compute_transformation(angle, 'y')

    # Rotate airfoil coordinates
    x = copy.deepcopy(element.x)
    z = copy.deepcopy(element.z)
    for j in range(len(element.x)):
        coords = np.array([[element.x[j]], [0.0], [element.z[j]]])

        rotated_coords = transformation_matrix.conj().transpose().dot(coords - centre_of_rotation) + centre_of_rotation

        x[j] = rotated_coords[0, 0]
        z[j] = rotated_coords[2, 0]
    return x, z



