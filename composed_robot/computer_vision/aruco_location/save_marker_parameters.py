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
from .markers import Markers
from ..camera_calibration.get_calibration_matrix import get_calibration_matrix
from pathlib import Path

# with open('camera_calibration_2/camera_cal_3.npy','rb') as f:
#     camera_matrix = np.load(f)
#     camera_distortion = np.load(f)
    
camera_matrix,camera_distortion = get_calibration_matrix()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
marker_size = 100

load_dotenv()

PI_IP = os.getenv("PI_IP")




time.sleep(1)

context = zmq.Context()
subscriber = Subscriber(PI_IP, [{'node':'vision',
                                  'topic':'frame'},
                                  ])

subscriber.start()

count = 0

markers = Markers()
count = 0
total = np.array([0,0,0])
reading_count = 0

rotation_matrices = []
translation_vectors = []

for (topic,node,bytes) in subscriber.bytes_stream():
    count+=1
    np_array = np.frombuffer(bytes,dtype=np.uint8)
    image = cv2.imdecode(np_array,1)
    
    image =  markers.get_markers(image)
    
    h,w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, (w,h), 1, (w,h))
    image = cv2.undistort(image, camera_matrix, camera_distortion, None, newcameramtx)
    x, y, w, h = roi
    image = image[y:y+h, x:x+w]
    
    
    cv2.imshow('image',image)

    if len(markers.markers)>0:
        marker = markers.markers[0]
        c_T_m = marker.c_T_m
        m_T_c = marker.m_T_c
        
        rotation_matrices.append(m_T_c.rotation_matrix)
        translation_vectors.append(m_T_c.translation_vector)
        
        
        
        
       
        if(count%15==0):
            reading_count+=1
            pitch,yaw,roll = m_T_c.get_pitch_yaw_roll()
            print(math.degrees(pitch),math.degrees(yaw),math.degrees(roll))
            
            # x,y,z,_ = m_T_c.homogeneous_matrix @ np.array([[0],[0],[0],[1]])
            x,y,z,_ = m_T_c.homogeneous_matrix @ np.array([0,0,0,1])
            
            
            #print(f'x: {x} y: {y} z: {z}')
            #print(f'x: {average[0]} y: {average[1]} z: {average[2]}')
        if count==300:
            break
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()

np.save(Path(__file__).parent/'rotation_matrices',rotation_matrices)
np.save(Path(__file__).parent/'translation_vectors',translation_vectors)


           
            








            




        

            
        

        




            
            








            




        

            
        

        




            
