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
from ...computer_vision.utilities.geometry import Pose

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

for (topic,node,bytes) in subscriber.bytes_stream():
    count+=1
    np_array = np.frombuffer(bytes,dtype=np.uint8)
    image = cv2.imdecode(np_array,1)
    image =  markers.get_markers(image)
    
    if(len(markers.markers) > 0):
        marker = markers.markers[0]
        c_T_m = marker.c_T_m
        m_T_c = marker.m_T_c
        w_T_m = marker.w_T_m
        
        w_HM_c = w_T_m.homogeneous_matrix@m_T_c.homogeneous_matrix
        w_T_c = Pose.create_from_homogeneous('w','c',w_HM_c)
        
        #x,y,z,_ =  m_T_c.homogeneous_matrix @ np.array([0,0,0,1])
        pitch,yaw,roll = w_T_c.get_pitch_yaw_roll()
        
        #from vector to go from location of camera in world to centre point between wheels of robot
        x_camera,y_camera,z,_ =  w_HM_c @ np.array([0,0,0,1])
        
        
        angle = (math.pi/2)+roll
        magnitude = 188.9552
        v = np.array([math.cos(angle)*magnitude,math.sin(angle)*magnitude])*-1
        
        
        position_of_camera = np.array([x_camera,y_camera])
        position_of_robot = position_of_camera+v
        x_robot,y_robot = position_of_robot
        
        
        
       
        if count%15==0:
            # print('roll',math.degrees(roll))
            # print({'x_camera': x_camera,'y_camera': y_camera,'z': z})
            print(x_robot,y_robot)
        count+=1
        
    cv2.imshow('not lost in translation',image)
    cv2.waitKey(1)
cv2.destroyAllWindows()

           
            








            




        

            
        

        




            
            








            




        

            
        

        




            
