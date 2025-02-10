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
from ..aruco_location.markers import Markers
from ..tyre_tracker.tyre_tracker import track_tyre
import math
from ..utilities.geometry import intersection_plane_and_line,get_normal_of_plane,to_homogenous,transform_vectors,drop_homogenous,point_to_homogenous


with open('camera_calibration_2/camera_cal_3.npy','rb') as f:
    camera_matrix = np.load(f)
    camera_distortion = np.load(f)
    


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
marker_size = 100

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

low = np.array([6,207,191])
high = np.array([15,255,255])

K_inv = np.linalg.inv(camera_matrix)

have_parameters = False

count = 0

markers = Markers()

f_x = camera_matrix[0,0]
f_y = camera_matrix[1,1]
o_x = camera_matrix[0,2]
o_y = camera_matrix[1,2]

def distance_along_z_axis_to_ground(c_T_m):
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


for (topic,node,bytes) in subscriber.bytes_stream():
    np_array = np.frombuffer(bytes,dtype=np.uint8)
    image = cv2.imdecode(np_array,1)
    markers.get_markers(image)
    centre = track_tyre(image,low,high)
   
    if len(markers.markers)>0 and centre:
        marker = markers.markers[0]
        c_T_m = marker.c_T_m
        m_T_c = marker.m_T_c
        
       
        if(count%15==0):
            # print('_____________')
            # print(c_T_m.homogeneous_matrix)
            # u = centre[0]
            # v = centre[1]
            # print(u,v)
            # print('_______________')
            u = centre[0]
            v = centre[1]
            c_P2_z = distance_along_z_axis_to_ground(c_T_m)
            #print('distance_along_z_axis_to_ground: ',c_P2_z)
            u = centre[0]
            v = centre[1]
            c_P2_x = (c_P2_z/f_x) * (u-o_x)
            c_P2_y = (c_P2_z/f_y) * (v-o_y)
            c_P2 = np.array([c_P2_x,c_P2_y,c_P2_z])
            m_points_on_m1 = np.array([[0,0,20],[1,0,20],[0,1,20]])
            
            # commented out points lie on the ground plane
            # m_points_on_m1 = np.array([[0,0,0],[1,0,0],[0,1,0]])
            # after correctly finding flat item on ground plane have raised the plane to top of lego
            #wheel by setting z to 20mm
            m_points_on_m1 = np.array([[0,0,20],[1,0,20],[0,1,20]])
            m_points_on_m1 = to_homogenous(m_points_on_m1)
            c_points_on_m1 = transform_vectors(c_T_m.homogeneous_matrix,m_points_on_m1)
            c_points_on_m1 = drop_homogenous(c_points_on_m1)
            c_normal_to_m1 = get_normal_of_plane(c_points_on_m1)
            c_point_on_m1 = c_points_on_m1[1]
            
            
            #line l
            c_point_on_line_l = np.array([0,0,0])#point on line
            c_vector_on_line_l = c_P2 - c_point_on_line_l #vector on line
            
            c_X = intersection_plane_and_line(c_point_on_m1,c_normal_to_m1,c_point_on_line_l,c_vector_on_line_l)
            m_X = transform_vectors(m_T_c.homogeneous_matrix,point_to_homogenous(c_X))
        
            print(m_X)
            
            
            
            
            
            
            
           
        count+=1  
    cv2.imshow('not lost in translation',image)
    if cv2.waitKey(1) == ord("q"):
        break
cv2.destroyAllWindows()
           

           
          
           
           


           

           
            








            



    
        

            
        

        




            
    

   