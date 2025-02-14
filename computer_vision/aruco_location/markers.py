from dotenv import load_dotenv

from robonet.Hub import Hub
from robonet.Subscriber import Subscriber
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2.aruco as aruco
from robonet.Publisher import Publisher
from ..utilities.geometry import Pose
from ..camera_calibration.get_calibration_matrix import get_calibration_matrix

    

camera_matrix,camera_distortion = get_calibration_matrix()

f_x = camera_matrix[0,0]
f_y = camera_matrix[1,1]
o_x = camera_matrix[0,2]
o_y = camera_matrix[1,2]


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
marker_size = 100


class Marker:
    def __init__(self,c_T_m,m_T_c,w_T_m,id):
        self.c_T_m = c_T_m
        self.m_T_c = m_T_c
        self.w_T_m = w_T_m
        self.id = id
        

class Markers:
    def __init__(self):
        self.image = None
        self.markers = []
        self.marker_locations = np.array([[0,0,0],[2003.8,0,0],[2003.8,1720.4,0],[0,1720.4,0],[730,2970,0]])


    def get_markers(self,image):
        self.markers = []
        self.image = image
        h,w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, (w,h), 1, (w,h))
        image = cv2.undistort(image, camera_matrix, camera_distortion, None, newcameramtx)
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]
   


        gray_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        # corners,ids,rejected = aruco.detectMarkers(gray_frame,aruco_dict,camera_matrix,camera_distortion)
        corners,ids,rejected = aruco.detectMarkers(gray_frame,aruco_dict,camera_matrix,0)
    
        if ids is not None:
            #added this line without checking effect as i noticed it was missing
           
            aruco.drawDetectedMarkers(image,corners)
        

            rotation_vectors,translation_vectors,_objPoints = aruco.estimatePoseSingleMarkers(corners,marker_size,camera_matrix,camera_distortion)

            for marker in range(len(ids)):
                marker_location_id = ids[marker][0]
                if marker_location_id>len(self.marker_locations)-1:
                    continue
                cv2.drawFrameAxes(image,camera_matrix, 0, rotation_vectors[marker], translation_vectors[marker],marker_size)
                c_rotation_vector_m = rotation_vectors[marker][0]
                c_rotation_matrix_m,_ = cv2.Rodrigues(c_rotation_vector_m)
                c_translation_vector_m = translation_vectors[marker][0]
                c_T_m = Pose('c','m',c_rotation_matrix_m,c_translation_vector_m)
                # print(c_T_m.homogeneous_matrix)
                m_T_c = c_T_m.get_reverse_pose()
                w_T_m = Pose('w','m',np.eye(3),self.marker_locations[marker_location_id])
                marker = Marker(c_T_m,m_T_c,w_T_m,marker_location_id)
                self.markers.append(marker)

                
        return image

    




