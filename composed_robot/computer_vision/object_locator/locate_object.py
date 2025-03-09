import numpy as np
from ..utilities.geometry import intersection_plane_and_line,get_normal_of_plane,to_homogenous,transform_vectors,drop_homogenous,point_to_homogenous


def locate_object(u,v,z,camera_matrix,c_T_d,d_T_c):
    """
    given the pixel co-ordinates to a point on a plane
    and the distance along the z axis to the plane this function gives us the location of the point on the plane in 3d coordinates in a desired co-ordinate system.
    
    u = horizontal pixel co-ordinate
    v = vertical pixel co-ordinate
    z = distance to the point in the real world along the camera optical axis
    camera_matrix : camera matrix obtained from camera calibration
    c_T_d : Pose from camera frame to d for desired frame we want the result in
    c_T_d : type : Pose (this is a class for working with poses in my computer vision library)
    d_T_c : reverse Pose of the previously described pose
    """
    
    f_x = camera_matrix[0,0]
    f_y = camera_matrix[1,1]
    o_x = camera_matrix[0,2]
    o_y = camera_matrix[1,2]
    
    c_P2_z = z
    
    c_P2_x = (c_P2_z/f_x) * (u-o_x)
    c_P2_y = (c_P2_z/f_y) * (v-o_y)
    c_P2 = np.array([c_P2_x,c_P2_y,c_P2_z])
    m_points_on_m1 = np.array([[0,0,8],[1,0,8],[0,1,8]])
    
    # commented out points lie on the ground plane
    # m_points_on_m1 = np.array([[0,0,0],[1,0,0],[0,1,0]])
    # after correctly finding flat item on ground plane have raised the plane to top of lego
    #wheel by setting z to 20mm
    m_points_on_m1 = np.array([[0,0,20],[1,0,20],[0,1,20]])
    m_points_on_m1 = to_homogenous(m_points_on_m1)
    c_points_on_m1 = transform_vectors(c_T_d.homogeneous_matrix,m_points_on_m1)
    c_points_on_m1 = drop_homogenous(c_points_on_m1)
    c_normal_to_m1 = get_normal_of_plane(c_points_on_m1)
    c_point_on_m1 = c_points_on_m1[1]
    #line l
    c_point_on_line_l = np.array([0,0,0])#point on line
    c_vector_on_line_l = c_P2 - c_point_on_line_l #vector on line
    
    c_X = intersection_plane_and_line(c_point_on_m1,c_normal_to_m1,c_point_on_line_l,c_vector_on_line_l)
    m_X = transform_vectors(d_T_c.homogeneous_matrix,point_to_homogenous(c_X))  
    return m_X    