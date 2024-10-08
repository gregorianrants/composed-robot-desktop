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
marker_size = 100

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

def get_pose(image):
    counter = 0
    gray_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    corners,ids,rejected = aruco.detectMarkers(gray_frame,aruco_dict,camera_matrix,camera_distortion)
    marker_is_found = False
    results = []

    if ids is not None:
        
        aruco.drawDetectedMarkers(image,corners)

        rotation_vectors,translation_vectors,_objPoints = aruco.estimatePoseSingleMarkers(corners,marker_size,camera_matrix,camera_distortion)

        for marker in range(len(ids)):
            marker_location_id = ids[marker][0]
            if marker_location_id>len(marker_locations)-1:
                continue
            cv2.drawFrameAxes(image,camera_matrix, camera_distortion, rotation_vectors[marker], translation_vectors[marker],100)

            #gives the rotation and translation of the marker in the camera frame
            rotation_vector = rotation_vectors[marker][0]
            translation_vector = translation_vectors[marker][0]

            #rotation and translation of the camera in the marker frame
            rotation_vec_camera_in_marker = rotation_vector*-1
            rotation_matrix_c_in_m,jacobian = cv2.Rodrigues(rotation_vec_camera_in_marker)

            pitch,roll,yaw = rotationMatrixToEulerAngles(rotation_matrix_c_in_m)
            
            #we flip the translation vector to get the translation of the camera relative to the marker frame
            #however the vector is still w.r.t. the camera
            translation_vec_camera_in_marker = translation_vector*-1  #w.r.t the camera
            translation_vec_camera_in_marker = np.dot(rotation_matrix_c_in_m,translation_vec_camera_in_marker) #w.r.t the camera
            tvec_str_in_marker = f"x={translation_vec_camera_in_marker[0]} y={translation_vec_camera_in_marker[1]} direction={math.degrees(yaw)}"

            

            

            translation_vec_camera_in_world = translation_vec_camera_in_marker + marker_locations[marker_location_id]

            tvec_str_in_world = f"x={translation_vec_camera_in_world[0]} y={translation_vec_camera_in_world[1]} direction={math.degrees(yaw)}"
            print(tvec_str_in_world)
            # if counter ==30:
            #     print(tvec_str_in_world)
            #     counter=0
            # counter+=1

            results.append([translation_vec_camera_in_world[0],translation_vec_camera_in_world[1],yaw])
    return (image,results)



