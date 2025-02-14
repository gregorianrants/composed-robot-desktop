import numpy as np
from ..utilities.geometry import intersection_plane_and_line,get_normal_of_plane,to_homogenous,transform_vectors,drop_homogenous,point_to_homogenous

def get_z(c_T_m):
    #-----------defining the ground plane -----------
    #points on plane w.r.t Marker
    #z axis has to be 0 x and y can be anything
    m_P = to_homogenous(np.array([[0,0,0],[0,1,0],[1,0,0]]))
    # points on plane w.r.t camera
    c_P = transform_vectors(c_T_m.homogeneous_matrix,m_P)
    c_P = drop_homogenous(c_P)
    normal_vector_of_plane = get_normal_of_plane(c_P)
    point_on_plane = c_P[0]
    
    #--------defining the line that goes through the optical axis
    #--------and interesects ground plane
    point_on_line_1 = np.array([0,0,0])
    point_on_line_2 = np.array([0,0,1])
    vector_along_line = point_on_line_2-point_on_line_1  
    
    #------------finding the intersection
    X = intersection_plane_and_line(point_on_plane,normal_vector_of_plane,point_on_line_1,vector_along_line)
    return np.linalg.norm(X)