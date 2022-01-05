# Robotics library
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
