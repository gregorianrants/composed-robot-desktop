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
from ...aruco_location.markers import Markers
from ...tyre_tracker.tyre_tracker import track_tyre
import math

from ...camera_calibration.get_calibration_matrix import get_calibration_matrix
from ..get_z import get_z
from ..locate_object import locate_object
from ...utilities.geometry import Pose
from ...camera_calibration.unwarp import unwarp



camera_matrix,distortion_matrix = get_calibration_matrix()



load_dotenv()

PI_IP = os.getenv("PI_IP")

time.sleep(1)

context = zmq.Context()
subscriber = Subscriber(PI_IP, [{'node':'vision',
                                  'topic':'frame'},
                                  ])

#print(camera_matrix)

subscriber.start()

low = np.array([15,226,239])
high = np.array([32,255,255])

low = np.array([10,205,221])
high = np.array([19,255,255])

K_inv = np.linalg.inv(camera_matrix)

have_parameters = False

count = 0

calibration_matrix = get_calibration_matrix()


m_T_c = Pose('m','c',np.array([[ 0.99890097, -0.02732713,  0.0380798 ],
       [-0.03925281, -0.93300009,  0.35780971],
       [ 0.02561367, -0.35887594, -0.93303723]]),np.array([ 121.02265276, -460.67071796,  556.57841212]))
c_T_m = m_T_c.get_reverse_pose()


for (topic,node,bytes) in subscriber.bytes_stream():
    np_array = np.frombuffer(bytes,dtype=np.uint8)
    image = cv2.imdecode(np_array,1)
    image = unwarp(image,calibration_matrix)
    
    centre = track_tyre(image,low,high)
   
    if centre:
        if(count%15==0):
            # print('_____________')
            # print(c_T_m.homogeneous_matrix)
            # u = centre[0]
            # v = centre[1]
            # print(u,v)
            # print('_______________')
            u = centre[0]
            v = centre[1]
            c_P2_z = get_z(c_T_m)
            #print('distance_along_z_axis_to_ground: ',c_P2_z)
            u = centre[0]
            v = centre[1]
            m_X = locate_object(u,v,c_P2_z,camera_matrix,c_T_m,m_T_c)
            
        
            print(m_X)
            
            
            
            
            
            
            
           
        count+=1  
    cv2.imshow('not lost in translation',image)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
           

           
          
           
           


           

           
            








            



    
        

            
        

        




            
    

   