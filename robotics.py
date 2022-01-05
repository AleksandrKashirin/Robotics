#!/usr/bin/env python

""" Robotics library
This file contains functions for calculation
of robotics manipulator kinematics and mechanics.
It is based on Denavit-Hartenberg notation.
Functions perform symbolic output using sympy library.
But it can also produce numerical outputs using "compute_num" function.
List of functions:
1. get_DH_table - Function returns Denavit-Hartenberg table
2. get_tf_matrix - Returns transformation matrix between i and i-1 frames
3. get_rotation_matrix - Returns rotation matrices between i and i-1 frames
4. get_displacement_vector - Returns displacement vector between i and i-1 frames
5. get_resulted_tf - Returns resulted transforamation matrix for end-effector
6. get_ef_pos - Returns end-effector position
7. get_res_rotation - Function returns resulted rotation matrix for specified DH-table
8. get_angular_velocity - Function returns angular velocity for current frame relative to the world-wide frame
9. cross - Function returns cross product for 3-row vectors
10. get_linear_velocity - Function returns linear velocity for current frame relative to the world-wide frame
11. get_ef_velocity - Function returns end effector velocity
12. jacobian - Function returns jacobian between two given vectors
13. calculate_static_force - Function returns a static force of the joint i
14. calculate_static_torque - Function calculates torque for the required joint in a static mode
15. compute_num - Function perform numerical computation
16. get_angular_acceleration - Function returns angular acceleration for the current joint
17. get_linear_acceleration - Function returns linear acceleration for the revolute joint
18. get_lin_acc_of_mass_center - Function returns linear acceleration of the center of mass of the link
19. calculate_dynamic_force - Function returns dynamic force in inward manner from last link to the base
20. calculate_dynamic_moment - Function calculates moment for the required joint in a dynamic mode
"""

__author__ = "Kashirin Aleksandr"
__copyright__ = "Copyright 2021, Public License"
__credits__ = ["Kashirin Aleksandr"]
__license__ = None
__version__ = "1.0.1"
__maintainer__ = "Kashirin Aleksandr"
__email__ = "Aleksandr.Kashirin@skoltech.ru"
__status__ = "In Progres"

import sympy as sp

# Define global variables
g = sp.symbols('g')  # g - Gravity constant
t = sp.symbols('t')  # t - Time variable

# Define library of functions
def get_DH_table(alpha, a, d, theta):
    """ Function returns Denavit-Hartenberg table

    Args:
        alpha - list of angles from Z_{i-1} to Z_i measured about X_{i-1} axis.
        a - list of distances from Z_{i-1} to Z_i measured along X_{i-1} axis.
        d - list is distances from X_{i-1} to X_i measured along Z_i axis.
        theta - list of angles from X_{i-1} to X_i measured about Z_i axis.

    Returns:
        DH_table - Denavit-Hartenberg table
    """
    return sp.Matrix([alpha, a, d, theta]).T

# Define the function to get transformation matrix
def get_tf_matrix(DH_table, i):
    """ Returns transformation matrix between i and i-1 frames

    Args:
        DH_table - Denavit-Hartenberg table of parameters
        i - Frame index

    Returns:
        T - Tranformation matrix
    """
    if (i - 1) >= 0:
        if ((i - 1) <= DH_table.shape[0]):

            # Assign the parameters
            alpha = DH_table[i - 1 , 0]
            a = DH_table[i - 1 , 1]
            d = DH_table[i - 1 , 2]
            theta = DH_table[i - 1 , 3]

            # Calculate transforamtion matrix
            T = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0, a],
                           [sp.sin(theta)*sp.cos(alpha), sp.cos(theta)*sp.cos(alpha), -sp.sin(alpha), -sp.sin(alpha)*d],
                           [sp.sin(theta)*sp.sin(alpha), sp.cos(theta)*sp.sin(alpha), sp.cos(alpha), sp.cos(alpha)*d],
                           [0, 0, 0, 1]])

            return sp.simplify(T)
        else:
            raise ValueError(f'Frame index should less or equal the maximum frame index. Maximum frame index: {DH_table.shape[0]}')
    else:
        raise ValueError('Frame index should be more than 0')

def get_rotation_matrix(DH_table, i):
    """ Returns rotation matrices between i and i-1 frames

    Args:
        DH_table - Denavit-Hartenberg table of parameters
        i - Frame index

    Returns:
        R - Rotation matrix
    """
    if (i - 1) >= 0:
        if ((i - 1) <= DH_table.shape[0]):

            # Assign the parameters
            alpha = DH_table[i - 1 , 0]
            theta = DH_table[i - 1 , 3]

            # Calculate rotation matrix
            R = sp.Matrix([[sp.cos(theta), -sp.sin(theta), 0],
                           [sp.sin(theta)*sp.cos(alpha), sp.cos(theta)*sp.cos(alpha), -sp.sin(alpha)],
                           [sp.sin(theta)*sp.sin(alpha), sp.cos(theta)*sp.sin(alpha), sp.cos(alpha)]])

            return sp.simplify(R)
        else:
            raise ValueError(f'Frame index should less or equal the maximum frame index. Maximum frame index: {DH_table.shape[0]}')
    else:
        raise ValueError('Frame index should be more than 0')

def get_displacement_vector(DH_table, i):
    """ Returns displacement vector between i and i-1 frames

    Args:
        DH_table - Denavit-Hartenberg table of parameters
        i - Frame index

    Returns:
        T - Tranformation matrix
    """
    if (i - 1) >= 0:
        if ((i - 1) <= DH_table.shape[0]):

            # Assign the parameters
            alpha = DH_table[i - 1 , 0]
            a = DH_table[i - 1 , 1]
            d = DH_table[i - 1 , 2]

            # Calculate displacement vector
            P = sp.Matrix([[a],
                           [-sp.sin(alpha)*d],
                           [sp.cos(alpha)*d]])

            return sp.simplify(P)
        else:
            raise ValueError(f'Frame index should less or equal the maximum frame index. Maximum frame index: {DH_table.shape[0]}')
    else:
        raise ValueError('Frame index should be more than 0')

def get_resulted_tf(DH_table):
    """ Returns resulted transforamation matrix for end-effector

    Args:
        DH_table - Denavit-Hartenberg table of parameters
    
    Returns:
        res_T - Resulted tranformation matrix
    """
    # Initialize list of all transforametion matrices
    res_T = 1
    for i in range(1, DH_table.shape[0] + 1):
        # Append new transforamtion matrix
        res_T *= get_tf_matrix(DH_table, i)
    return sp.trigsimp(res_T)

def get_ef_pos(DH_table):
    """Returns end-effector position

    Args:
        DH_table - Denavit-Hartenberg table of parameters
    
    Returns:
        pos - End-effector position
    """
    pos = get_resulted_tf(DH_table)[:3, 3]
    return sp.trigsimp(pos)

def get_res_rotation(DH_table):    
    """ Function returns resulted rotation matrix for specified DH-table

    Args:
        DH_table - Denavit-Hartenberg table of parameters

    Returns:
        rot - Resulted rotation matrix
    """
    rot = get_resulted_tf(DH_table)[:3, :3]
    return sp.trigsimp(rot)

def get_angular_velocity(theta_dots, i, DH_table):
    """ Function returns angular velocity for current frame
        relative to the world-wide frame

    Args:
        theta_dots - Array of joint angular velocities
        i - Index of frame for which we want to find angular velocity
        DH_table - Denavit-Hartenberg table of parameters

    Returns:
        w_i - Current angular velocity relative to the world frame
    """
    # Define initial velocity
    w_0_0 = sp.Matrix([0, 0, 0])
    # Initialize vector Z
    Z = sp.Matrix([0, 0, 1])
    # Initialize list of angular velocities
    W = [w_0_0]
    if i == 0:
        w_i = w_0_0
    # Check if number of the last frame is less than
    # Amount of joint velocities in an array
    elif i <= len(theta_dots):
        # For each joint
        for j in range(1, i + 1):
            # Calculate R
            R_transpose = get_rotation_matrix(DH_table, j)
            # Transpose the R
            R = R_transpose.T
            # Calculate w_i
            w_i = R * W[j - 1] + theta_dots[j - 1] * Z
            # Append new speed to the array
            W.append(w_i)
    else:
        raise ValueError('n is out of range! Check the amount of frames.')
    return w_i

def cross(a, b):
    """ Function returns cross product for 3-row vectors

    Args:
        a - Vector - column
        b - Vector - column
    
    Return:
        res - Cross product result
    """
    # Assign the variables
    # Vector a
    x1 = a[0]
    y1 = a[1]
    z1 = a[2]
    # Vector b
    x2 = b[0]
    y2 = b[1]
    z2 = b[2]
    # Calculate result
    res = sp.Matrix([[y1*z2 - y2*z1],
                     [z1*x2 - z2*x1],
                     [x1*y2 - x2*y1]])
    return res

def get_linear_velocity(theta_dots, i, DH_table):
    """ Function returns linear velocity for current frame
        relative to the world-wide frame

    Args:
        theta_dots - Array of joint angular velocities
        n - Index of frame for which we want to find angular velocity
        P - Array of displacement vectors

    Returns:
        w_i - Current angular velocity relative to the world frame
    """
    # Define initial velocity
    v_0_0 = sp.Matrix([0, 0, 0])
    # Initialize vector Z
    Z = sp.Matrix([0, 0, 1])
    # Initialize list of angular velocities
    V = [v_0_0]
    if i == 0:
        v_i = v_0_0
    # Check if number of the last frame is less than
    # Amount of joint velocities in an array
    elif i <= len(theta_dots):
        # For each joint
        for j in range(1, i + 1):
            # Calculate R
            R_transpose = get_rotation_matrix(DH_table, j)
            # Transpose the R
            R = R_transpose.T
            # Calculate P
            P = get_displacement_vector(DH_table, j)
            # Calculate angular velocity w_{i-1}
            w = get_angular_velocity(theta_dots, j - 1, DH_table)
            # Calculate linear velocity v_i
            v_i = R * (V[j - 1] + cross(w, P))
            # Append new speed to the array
            V.append(v_i)
    else:
        raise ValueError('n is out of range! Check the amount of frames.')
    return v_i

def get_ef_velocity(theta_dots, DH_table, method='disp'):
    """ Function returns end effector velocity

    Args:
        theta_dots - Array of joint angular velocities
        DH_table - Denavit-Hartenberg table of parameters
        method - Type of methodology to calculate velocity:
                rot - from rotational matrix   [This feature doesn't work]
                disp - from displacement vector

    Returns:
        res_v - end-effector linear velocity
    """
    if method == 'rot':
        res_v = get_res_rotation(DH_table) @ get_linear_velocity(theta_dots, len(theta_dots))
    elif method == 'disp':
        res_v = sp.diff(get_ef_pos(DH_table), t)
    return sp.simplify(res_v)

def jacobian(a, b):
    """ Function returns jacobian between two given vectors

    Args:
        a - Vector a
        b - Vector b
    
    Returns:
        J - Jacobian of two vectors
    """
    # Get length of vectors
    n = len(a)
    m = len(b)
    # Initialize Jacobian
    J = sp.zeros(n, m)
    # For each cell in Jacobian
    for i in range(n):
        for j in range(m):
            # Get derivative with related respect
            J[i, j] = a[i].diff(b[j])
    return sp.simplify(J)

def calculate_static_force(f_ef, i, DH_table):
    """ Function returns a static force of the joint i if f_ef was applied at the end-effector

    Args:
        f_ef - Force applied at the end-effector
            Note, that this force always applied to the last link,
            hence F_n = f_ef
        i - Index of the torque we want to calculate, should be more
            than 0 and less or equal n (total amount of joints)
        DH_table - Denavit-Hartenberg table of parameters

    Returns:
        F[:, i - 1] - Calculated force vector of the required joint
    """
    # Define amount of possible forces to calculate
    n = DH_table.shape[0]
    # Initialize array of forces
    F = sp.zeros(3, n)
    # Define the last end-effector force
    F[:, -1] = f_ef
    # Check requested frame index
    if (i > 0) and (i <= n):
        # Calculate each force in backward direction
        for j in range(n - 2, -1, -1):
            # Calculate rotation matrix
            R = get_rotation_matrix(DH_table, j + 2)
            # Calculate Force
            f = R * F[:, j + 1]
            # Append new calculated force
            F[:, j] = f
        # Return requested force
        return sp.expand(sp.simplify(F[:, i - 1]))
    else:
        raise ValueError(f'Index frame is out of range, i should be > 0 and i <= {n}')

def calculate_static_torque(f_ef, i, DH_table, tau_ef=sp.Matrix([0, 0, 0])):
    """ Function calculates torque for the required joint in a static mode
        if f_ef was applied at the end-effector

    Args:
        f_ef - Force applied at the end-effector
            Note, that this force always applied to the last link,
            hence F_n = f_ef
        i - Index of the torque we want to calculate, should be more
            than 0 and less or equal n (total amount of joints)
        DH_table - Denavit-Hartenberg table of parameters
        tau_ef - Applied torque at the end effector, usually equals to zero

    Returns:
        T[:, i - 1] - Calculated torque vector of the required joint
    """
    # Define amount of possible torques to calculate
    n = DH_table.shape[0]
    # Initialize array of torques
    T = sp.zeros(3, n)
    # Define the last end-effector force
    T[:, -1] = tau_ef
    # Check requested frame index
    if (i > 0) and (i <= n):
        # Calculate each force in backward direction
        for j in range(n - 2, -1, -1):
            # Calculate rotation matrix
            R = get_rotation_matrix(DH_table, j + 2)
            # Calculate the displacement vector
            P = get_displacement_vector(DH_table, j + 2)
            # Calculate force
            f = calculate_static_force(f_ef, j + 1)
            # Calculate cross product
            prod = cross(P, f)
            # Calculate torque
            tau = R * T[:, j + 1] + prod
            # Append calculated torque
            T[:, j] = tau
        # Return requested torque
        return sp.expand(sp.simplify(T[:, i - 1]))
    else:
        raise ValueError(f'Index frame is out of range, i should be > 0 and i <= {n}')

def compute_num(func, params, num_params):
    """ Function perform numerical computation
    
    Args:
        func - Given function or Symbolic matrix to calculate
        params - tuple or list of parameters used in the function
        num_params - Corresponded numerical values of the parameters

    Returns:
        func_num - Numerical computed values of the function
    """
    # Lambdify the given function
    func_num = sp.lambdify(params, func)
    # Calculate numerical values
    func_num = func_num(*num_params)
    return func_num

def get_angular_acceleration(theta_dots, i, DH_table, joint='revolute', w_dot_0_0=sp.Matrix([0, 0, 0])):
    """ Function returns angular acceleration for the current joint
    
    Args:
        theta_dots - Array of joint angular velocities
        i - Index of frame for which we want to find angular acceleration
        joint - Type of joint ['revolute', 'prismatic']
        DH_table - Denavit-Hartenberg table of parameters
        w_dot_0_0 - Angular acceleration of the base of a robot

    Returns:
        w_dot_i - Current angular acceleration relative to the world frame

    """
    # Initialize array of angular accelerations
    theta_dots_dots = sp.zeros(len(theta_dots), 1)
    for k in range(len(theta_dots)):
        theta_dots_dots[k] = theta_dots[k].diff()
    # Initialize vector Z
    Z = sp.Matrix([0, 0, 1])
    # Initialize list of angular accelerations
    W_dot = [w_dot_0_0]
    if i == 0:
        w_dot_i = w_dot_0_0
    # Check if number of the last frame is less than
    # Amount of joint velocities in an array
    elif i <= len(theta_dots):
        # For each joint
        for j in range(1, i + 1):
            # Calculate R
            R_transpose = get_rotation_matrix(DH_table, j)
            # Transpose the R
            R = R_transpose.T
            # Calculate angular velocity
            w_i = get_angular_velocity(theta_dots, j - 1, DH_table=DH_table)
            # Pick the joint
            if joint == 'revolute':
                # Calculate cross product
                cross_prod = cross(R * w_i, theta_dots[j - 1] * Z)
                # Calculate angular acceleration
                w_dot_i = R * W_dot[j - 1] + cross_prod + theta_dots_dots[j - 1] * Z
            elif joint == 'prismatic':
                # Calculate angular acceleration
                w_dot_i = R * w_i
            # Append new speed to the array
            W_dot.append(w_dot_i)
    else:
        raise ValueError('n is out of range! Check the amount of frames.')
    return sp.expand(sp.simplify(w_dot_i))

def get_linear_acceleration(theta_dots, i, DH_table, v_dot_0_0=sp.Matrix([0, g, 0])):
    """ Function returns linear acceleration for the revolute joint

    Args:
        theta_dots - Array of joint angular velocities
        i - Index of frame for which we want to find angular acceleration
        DH_table - Denavit-Hartenberg table of parameters
        v_dot_0_0 - Linear acceleration of the base of a robot

    Returns:
        v_dot_i - Current linear acceleration relative to the world frame
    """
     # Initialize vector Z
    Z = sp.Matrix([0, 0, 1])
    # Initialize list of linear accelerations
    V_dot = [v_dot_0_0]
    if i == 0:
        v_dot_i = v_dot_0_0
    # Check if number of the last frame is less than
    # Amount of joint velocities in an array
    elif i <= len(theta_dots):
        # For each joint
        for j in range(1, i + 1):
            # Calculate R
            R_transpose = get_rotation_matrix(DH_table, j)
            # Transpose the R
            R = R_transpose.T
            # Calculate angular velocity
            w_i = get_angular_velocity(theta_dots, j - 1, DH_table)
            # Calculate angular acceleration
            w_dot_i = get_angular_acceleration(theta_dots, j - 1, DH_table=DH_table, joint='revolute')
            # Get displacement vector
            P = get_displacement_vector(DH_table, j)
            # Calculate linear acceleration
            v_dot_i = R * (cross(w_dot_i, P) + cross(w_i, cross(w_i, P)) + V_dot[j - 1])
            # Append new speed to the array
            V_dot.append(v_dot_i)
    else:
        raise ValueError('n is out of range! Check the amount of frames.')
    return sp.expand(sp.simplify(v_dot_i))

def get_lin_acc_of_mass_center(theta_dots, i, P_c, DH_table):
    """ Function returns linear acceleration of the center of mass of the link

    Args:
        theta_dots - Array of joint angular velocities
        i - Index of frame for which we want to find angular acceleration
        P_c - Array of points that describes center of mass for each link [3xn], where n is amount mass centers
        DH_table - Denavit-Hartenberg table of parameters
        v_c_dot_0_0 - Linear acceleration of the base of a robot
    
    Returns:
        v_c_dot_i - Linear acceleration of the current link of the center of mass
    """
    # Check the amount of center origins in P_c 
    if P_c.shape[1] != (theta_dots.shape[0] - 1):
        raise ValueError(f'Not all centers of masses were specified in P_c. Amount of centers specified: {P_c.shape[1]}. Required amount of centers for links: {theta_dots.shape[0] - 1}')
    # Initialize vector Z
    Z = sp.Matrix([0, 0, 1])
    # Calculate linear acceleration
    if i == 0:
        raise ValueError(f'Index should be: >0 and <= {P_c.shape[1]}. Entered index: {i}')
    else:
        # Get angular velocity
        w_i = get_angular_velocity(theta_dots, i, DH_table=DH_table)
        # Get angular acceleration
        w_dot_i = get_angular_acceleration(theta_dots, i, joint='revolute', DH_table=DH_table)
        # Get linear acceleration
        v_dot_i = get_linear_acceleration(theta_dots, i, DH_table=DH_table)
        # Calculate linear acceleration of the center of mass
        v_c_dot_i = cross(w_dot_i, P_c[:, i - 1]) + cross(w_i, cross(w_i, P_c[:, i - 1])) + v_dot_i
        return sp.expand(sp.simplify(v_c_dot_i))

def calculate_dynamic_force(m, i, v_c_dot_i, DH_table):
    """ Function returns dynamic force in inward manner from last link to the base

    Args:
        m - Array of links masses
        i - Index of the force we want to calculate, should be more
            than 0 and less or equal n (total amount of joints)
        v_c_dot_i - Linear acceleration of the current link of the center of mass
        DH_table - Denavit-Hartenberg table of parameters

    Returns:
        f_i - Calculated force at the joint
    """
    # Define amount of possible forces to calculate
    n = DH_table.shape[0]
    # Check requested frame index
    if (i > 0) and (i <= n):
        # Calculate force
        F = m * v_c_dot_i
        # Return requested force
        return sp.expand(sp.simplify(F))
    else:
        raise ValueError(f'Index frame is out of range, i should be > 0 and i <= {n}')

def calculate_dynamic_moment(I, i, theta_dots, DH_table):
    """ Function calculates moment for the required joint in a dynamic mode

    Args:
        I - Moment of inertia for the link
        i - Index of the force we want to calculate, should be more
            than 0 and less or equal n (total amount of joints)
        theta_dots - Array of joint angular velocities
        DH_table - Denavit-Hartenberg table of parameters

    Returns:
        T[:, i - 1] - Calculated torque vector of the required joint
    """
    # Define amount of possible torques to calculate
    n = DH_table.shape[0]
    # Check requested frame index
    if (i > 0) and (i <= n):
        # Get angular acceleration
        w_dot = get_angular_acceleration(theta_dots, i, joint='revolute', DH_table=DH_table)
        # Get angular velocity
        w = get_angular_velocity(theta_dots, i, DH_table)
        # Calculate moment
        N = I * w_dot + cross(w, (I * w))
        # Return requested torque
        return sp.expand(sp.simplify(N))
    else:
        raise ValueError(f'Index frame is out of range, i should be > 0 and i <= {n}')