import numpy as np
import math

def intersection_plane_and_line(point_on_plane,normal_to_plane,point_on_line,vector_along_line):
  # all points and vectors are given as 1d arrays, (dont get dimensions of array and vector mixed up)
  # 3d array can represented as (,3) array or (3,1) array both are 3d vectors.
  q = point_on_plane
  n = normal_to_plane
  p = point_on_line
  v = vector_along_line
  t= np.dot((q-p),n)/np.dot(v,n)
  print(t)
  x = p + t*v
  return x

def get_normal_of_plane(points):
    p1,p2,p3 = points
    v1 = p3-p1
    v2 = p3-p2
    n = np.cross(v1,v2)
    return n/np.linalg.norm(n)
  
def transform_vectors(M,vectors_list):
    #applies a matrix transformation to each vector in a list of vectors
    return (M@vectors_list.T).T
  
def to_homogenous(input_points):
    h_output_points = np.ones((input_points.shape[0],4))
    h_output_points[:,:3]=input_points
    return h_output_points
  
def drop_homogenous(M):
  return M[:,:3]

def point_to_homogenous(point):
  result = np.array([0,0,0,1])
  result[0:3]=point.copy()
  return result

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
  
  
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])
  
class Pose:
    #comments work thought starting off with pose c_T_m and reverse pose returning m_T_c
    def __init__(self,move_from,move_to,rotation_matrix,translation_vector):
        self.move_from = move_from #c
        self.move_to = move_to  #m
        self.wrt = move_from #c
        self.rotation_matrix = rotation_matrix # c_R_m
        self.homogeneous_rotation = None
        self.translation_vector = translation_vector # c_tv_m
        self.homogeneous_matrix = self.create_homogeneous_matrix()
        
    def create_homogeneous_matrix(self):
        h_translation = np.eye(4)
        h_translation[:3,3]=self.translation_vector
        h_rotation = np.eye(4)
        h_rotation[0:3,0:3]=self.rotation_matrix
        self.homogeneous_rotation = h_rotation
        result = h_translation@h_rotation
        return result
    
    def get_pitch_yaw_roll(self):
        pitch,yaw,roll =  rotationMatrixToEulerAngles(self.rotation_matrix)
        return pitch,yaw,roll
    
    def get_reverse_pose(self):
        #start with translation c_tv_m wrt camera
        # m_tv_c_wrt_c
        translation_vector = -1*self.translation_vector 
        #     m_tv_c_wrt_m        (c_R_m.T=m_R_c)                m_tv_c_wrt_c
        translation_vector = self.rotation_matrix.T @ translation_vector
        #    m_r_c             c_R_m
        rotation_matrix = self.rotation_matrix.T
        return Pose(move_from=self.move_to,move_to=self.move_from,rotation_matrix=rotation_matrix,translation_vector=translation_vector)
 