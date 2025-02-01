
from dotenv import load_dotenv
import os
from robonet.Hub import Hub
from robonet.Subscriber import Subscriber
import zmq
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2.aruco as aruco
import math
from robonet.Publisher import Publisher

with open('camera_cal.npy','rb') as f:
    camera_matrix = np.load(f)
    camera_distortion = np.load(f)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
marker_size = 9.2/100

# def isRotationMatrix(R) :
#     Rt = np.transpose(R)
#     shouldBeIdentity = np.dot(Rt, R)
#     I = np.identity(3, dtype = R.dtype)
#     n = np.linalg.norm(I - shouldBeIdentity)
#     return n < 1e-6
 
# # Calculates rotation matrix to euler angles
# # The result is the same as MATLAB except the order
# # of the euler angles ( x and z are swapped ).
# def rotationMatrixToEulerAngles(R) :
 
#     assert(isRotationMatrix(R))
 
#     sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
#     singular = sy < 1e-6
 
#     if  not singular :
#         x = math.atan2(R[2,1] , R[2,2])
#         y = math.atan2(-R[2,0], sy)
#         z = math.atan2(R[1,0], R[0,0])
#     else :
#         x = math.atan2(-R[1,2], R[1,1])
#         y = math.atan2(-R[2,0], sy)
#         z = 0
 
#     return np.array([x, y, z])


# counter = 0

# marker_locations = np.array([[0,0,0],[2003.8,0,0],[2003.8,1720.4,0],[0,1720.4,0],[730,2970,0]])



# # marker_locations = {
# #     106: np.array([2003.8,1720.4])
# # }

# class Marker:
#     def __init__(self,rotation_vector,translation_vector,corners,marker_location_id):
#         self.rotation_matrix_m_to_c = None
#         self.rotation_vector_camera_to_marker = rotation_vector
#         self.translation_vector_camera_to_marker = translation_vector
#         self.counter = 0
#         self.pose_m_to_c = None
#         self.pose_c_to_m = None
#         self.pose_c_to_m_scratch = None
#         self.corners = corners
#         self.marker_location_id = marker_location_id

#     #this is used internally and works with non homogenous points
#     def _change_basis_camera_to_marker(self,wrt_camera):
#         wrt_marker = np.dot(self.rotation_matrix_m_to_c,wrt_camera)
#         return wrt_marker
    
#     def get_pitch_roll_yaw(self):
#         pitch,yaw,roll =  rotationMatrixToEulerAngles(self.rotation_matrix_m_to_c)
#         return pitch,yaw,roll
    
#     def get_homogenous_pose(self,translation_vec,rotation_matrix):
#         h_translation = np.eye(4)
#         h_translation[:3,3]=translation_vec
#         h_rotation = np.eye(4)
#         h_rotation[0:3,0:3]=rotation_matrix
#         result = h_translation@h_rotation
#         return result
    
#     def get_reverse_pose(self,translation_vec,rotation_matrix):
#         h_translation = np.eye(4)
#         h_translation[:3,3]=-1*translation_vec
#         h_rotation = np.eye(4)
#         h_rotation[0:3,0:3]=rotation_matrix.T
#         result = h_rotation@h_translation
#         return result
    
#     def handle_geometry(self):
#         #we are passed the rotation and translation of the marker in the camera frame
#         #rotation and translation of the camera in the marker frame
#         rotation_vec_marker_to_camera = self.rotation_vector_camera_to_marker*-1
#         self.rotation_matrix_m_to_c,jacobian = cv2.Rodrigues(rotation_vec_marker_to_camera)
#         #we flip the translation vector to get the translation of the camera relative to the marker frame
#         #however the vector is still w.r.t. the camera
#         self.translation_vec_marker_to_camera = np.dot(self.rotation_matrix_m_to_c,self.translation_vector_camera_to_marker*-1 ) #w.r.t the marker
#         self.pose_m_to_c = self.get_homogenous_pose(self.translation_vec_marker_to_camera,self.rotation_matrix_m_to_c)
#         self.pose_c_to_m = self.get_reverse_pose(self.translation_vec_marker_to_camera,self.rotation_matrix_m_to_c)
        
  

# class Pose:
#     def __init__(self):
#         self.image = None
#         self.markers = []


#     def get_pose(self,image):
#         self.image = image


#         gray_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         corners,ids,rejected = aruco.detectMarkers(gray_frame,aruco_dict,camera_matrix,camera_distortion)
    
#         if ids is not None:
#             #added this line without checking effect as i noticed it was missing
#             self.markers = []
#             aruco.drawDetectedMarkers(image,corners)
        

#             rotation_vectors,translation_vectors,_objPoints = aruco.estimatePoseSingleMarkers(corners,marker_size,camera_matrix,camera_distortion)

#             for marker in range(len(ids)):
#                 marker_location_id = ids[marker][0]
#                 if marker_location_id>len(marker_locations)-1:
#                     continue
#                 cv2.drawFrameAxes(image,camera_matrix, camera_distortion, rotation_vectors[marker], translation_vectors[marker],marker_size)


#                 rotation_vector = rotation_vectors[marker][0]
#                 translation_vector = translation_vectors[marker][0]
#                 marker = Marker(rotation_vector,translation_vector,corners[marker][0],marker_location_id)
#                 self.markers.append(marker)
#                 marker.handle_geometry()
#         return image

from spatialmath import SE3


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


counter = 0

marker_locations = np.array([[0,0,0],[2003.8,0,0],[2003.8,1720.4,0],[0,1720.4,0],[730,2970,0]])



# marker_locations = {
#     106: np.array([2003.8,1720.4])
# }

class Marker:
    def __init__(self,rotation_vector,translation_vector,corners,marker_location_id):
        self.rotation_matrix_m_to_c = None
        self.rotation_vector_camera_to_marker = rotation_vector
        self.translation_vector_camera_to_marker = translation_vector
        self.counter = 0
        self.pose_m_to_c = None
        self.pose_c_to_m = None
        self.pose_c_to_m_scratch = None
        self.corners = corners
        self.marker_location_id = marker_location_id

    #this is used internally and works with non homogenous points
    def _change_basis_camera_to_marker(self,wrt_camera):
        wrt_marker = np.dot(self.rotation_matrix_m_to_c,wrt_camera)
        return wrt_marker
    
    def get_pitch_roll_yaw(self):
        pitch,yaw,roll =  rotationMatrixToEulerAngles(self.rotation_matrix_m_to_c)
        return pitch,yaw,roll
    
    def get_homogenous_pose(self,translation_vec,rotation_matrix):
        h_translation = np.eye(4)
        h_translation[:3,3]=translation_vec
        h_rotation = np.eye(4)
        h_rotation[0:3,0:3]=rotation_matrix
        result = h_translation@h_rotation
        return result
    
    def get_reverse_pose(self,translation_vec,rotation_matrix):
        h_translation = np.eye(4)
        h_translation[:3,3]=-1*translation_vec
        h_rotation = np.eye(4)
        h_rotation[0:3,0:3]=rotation_matrix.T
        result = h_rotation@h_translation
        return result
    
    def handle_geometry(self):
        #we are passed the rotation and translation of the marker in the camera frame
        #rotation and translation of the camera in the marker frame
        rotation_vec_marker_to_camera = self.rotation_vector_camera_to_marker*-1
        self.rotation_matrix_m_to_c,jacobian = cv2.Rodrigues(rotation_vec_marker_to_camera)
        #we flip the translation vector to get the translation of the camera relative to the marker frame
        #however the vector is still w.r.t. the camera
        
        translation_vec_marker_to_camera = self.translation_vector_camera_to_marker*-1 #wrt camera
        self.translation_vec_marker_to_camera = np.dot(self.rotation_matrix_m_to_c,self.translation_vector_camera_to_marker) #w.r.t the marker
        # self.translation_vec_marker_to_camera = np.dot(self.rotation_matrix_m_to_c,self.translation_vector_camera_to_marker*-1 ) #w.r.t the marker
        self.pose_m_to_c = self.get_homogenous_pose(self.translation_vec_marker_to_camera,self.rotation_matrix_m_to_c)
        self.pose_c_to_m = self.get_reverse_pose(self.translation_vec_marker_to_camera,self.rotation_matrix_m_to_c)
        
class Pose:
    #comments work thought starting off with pose c_T_m and reverse pose returning m_T_c
    def __init__(self,move_from,move_to,rotation_matrix,translation_vector):
        self.move_from = move_from #c
        self.move_to = move_to  #m
        self.wrt = move_from #c
        self.rotation_matrix = rotation_matrix # c_R_m
        self.translation_vector = translation_vector # c_tv_m
        self.homogeneous_matrix = self.create_homogeneous_matrix()
        
    def create_homogeneous_matrix(self):
        h_translation = np.eye(4)
        h_translation[:3,3]=self.translation_vector
        h_rotation = np.eye(4)
        h_rotation[0:3,0:3]=self.rotation_matrix
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
        
        
class Markers:
    def __init__(self):
        self.image = None
        self.markers = []
        self.peters_markers = []


    def get_markers(self,image):
        self.image = image


        gray_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        corners,ids,rejected = aruco.detectMarkers(gray_frame,aruco_dict,camera_matrix,camera_distortion)
    
        if ids is not None:
            #added this line without checking effect as i noticed it was missing
            self.markers = []
            aruco.drawDetectedMarkers(image,corners)
        

            rotation_vectors,translation_vectors,_objPoints = aruco.estimatePoseSingleMarkers(corners,marker_size,camera_matrix,camera_distortion)

            for marker in range(len(ids)):
                marker_location_id = ids[marker][0]
                if marker_location_id>len(marker_locations)-1:
                    continue
                cv2.drawFrameAxes(image,camera_matrix, camera_distortion, rotation_vectors[marker], translation_vectors[marker],marker_size)
                c_rotation_vector_m = rotation_vectors[marker][0]
                c_rotation_matrix_m,_ = cv2.Rodrigues(c_rotation_vector_m)
                c_translation_vector_m = translation_vectors[marker][0]
                c_T_m = Pose('c','m',c_rotation_matrix_m,c_translation_vector_m)
                print(c_T_m.homogeneous_matrix)
                m_T_c = c_T_m.get_reverse_pose()
                self.markers.append(m_T_c)
                
                #p_c_T_m = SO3.RTvec(rotation_vectors[marker][0],translation_vectors[marker][0])
                p_c_T_m = SE3.Rt(c_rotation_matrix_m,translation_vectors[marker][0])
                print(p_c_T_m)
                
                # self.peters_markers.append(p_m_T_c)
                
        return image





