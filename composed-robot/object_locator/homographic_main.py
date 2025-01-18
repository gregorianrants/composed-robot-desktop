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
from .aruco_location import Pose
from .tyre_tracker import track_tyre
import math
from sympy import Plane,Line3D

with open('camera_cal.npy','rb') as f:
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

K_inv = np.linalg.inv(camera_matrix)

have_parameters = False

count = 0

pose = Pose()

def normal_of_plane(points):
    p1,p2,p3 = points.T
    v1 = p3-p1
    v2 = p3-p2
    n = np.cross(v1,v2)
    return n/np.linalg.norm(n)

def intersection_of_plane_and_line(point_on_plane,normal_vector_of_plane,point_on_line,vector_on_line):
    q = point_on_plane
    n = normal_vector_of_plane
    p = point_on_line
    v = vector_on_line
    X = p + ( (q-p).dot(n)/(v.dot(n)) )*v
    return X

#turns a matrix containing points in columns into homogenous points
#dont cnfuse with convertint a transformation matrix
def to_homogenous(input_matrix):
    h_input_matrix = np.ones((4,3))
    h_input_matrix[:3,:]=input_matrix
    return h_input_matrix


def drop_homogenous(M):
  return M[:3,:]


def get_z(pose_c_to_m):
    points_ground_wrt_marker = np.array([[0,0,0],[1,0,0],[0,1,0]]).T
    points_ground_wrt_camera = pose_c_to_m @ to_homogenous(points_ground_wrt_marker)
    #print('asdfsadf',points_ground_wrt_camera)
    points_ground_wrt_camera = drop_homogenous(points_ground_wrt_camera)
    #print(points_ground_wrt_camera)
    n = normal_of_plane(points_ground_wrt_camera)

    point_on_z_1 = np.array([0,0,0])
    point_on_z_2 = np.array([0,0,1])
    vector_on_line = point_on_z_2-point_on_z_1
    #print('vol',vector_on_line)
    point_on_plane = points_ground_wrt_camera.T[0]
    #print('ppp',point_on_plane)

    z_vec = intersection_of_plane_and_line(point_on_plane,n,point_on_z_2,vector_on_line)
    #print(z_vec)

    z_length = np.linalg.norm(z_vec)

    return z_length,z_vec
        
def find_X_legacy():
    U = np.array([centre[0],centre[1],z_length])
    P_c = K_inv.dot(U)
    rotation_matrix_c_to_m = marker.rotation_matrix_m_to_c.T
    plane_parralel_to_camera = np.array([[0,0,0],[0,1,0],[1,1,0]]).T
    plane_parralel_to_ground = rotation_matrix_c_to_m @ plane_parralel_to_camera
    z_vec = np.expand_dims(z_vec,axis=1)
    plane_parralel_to_ground = plane_parralel_to_ground + z_vec
    q1 = plane_parralel_to_ground.T[0]
    q2 = plane_parralel_to_ground.T[1]
    q3 = plane_parralel_to_ground.T[2]
    q = q1

    w1 = q3-q2
    w2 = q3-q1
    n = np.cross(w1,w2)


    point_on_line_1 = np.array([0,0,0])
    point_on_line_2 = P_c
    v = point_on_line_2-point_on_line_1
    p = point_on_line_1


    X = p + ( (q-p).dot(n)/(v.dot(n)) )*v
    X = np.expand_dims(np.hstack((X,[1])),axis=1)
    X = marker.pose_m_to_c @ X
    x,y,z,_ = X
    print(X)
    print(f'x: {x[0]*100} y: {y[0]*100} z: {z[0]*100}')

def point_to_homogenous(point):
    return np.expand_dims(np.hstack((point,[1])),axis=1)

def point_drop_homogenous(point):
    point = point/point[-1]
    return point[0:3]



f_x = camera_matrix[0,0]
f_y = camera_matrix[1,1]
o_x = camera_matrix[0,2]
o_y = camera_matrix[1,2]


for (topic,node,bytes) in subscriber.bytes_stream():
    np_array = np.frombuffer(bytes,dtype=np.uint8)
    image = cv2.imdecode(np_array,1)
    centre = track_tyre(image,low,high)
    image =  pose.get_pose(image)

    


    if len(pose.markers)>1 and centre:
        
        
        # pitch,roll,yaw = marker.get_pitch_roll_yaw()
        if(count%15==0):
            # marker = pose.markers[0]
            # print(marker.corners)
            # marker_coords = np.array([[-5,-5],[5,-5],[5,5],[-5,5]])
            # marker_coords = np.array([[-4.5,-4.5],[4.5,-4.5],[4.5,4.5],[-4.5,4.5]])/100
            # H = cv2.findHomography(marker.corners.copy(),marker_coords)[0]
            # centre_h = point_to_homogenous(centre)
            # #print(H)
            # x,y,_ = (H @ centre_h)
            # print(f'x: {x}, y:{y}')

            
            if pose.markers[0].marker_location_id < pose.markers[1].marker_location_id:
                m0 = pose.markers[0]
                m1 = pose.markers[1]
            else: 
                m0 = pose.markers[1]
                m1 = pose.markers[0]

            pose_m0_m1 = m0.pose_m_to_c @ m1.pose_c_to_m

            #real world centre locations of markers
            m0_centre = np.array([0,0,0,1])
            
            #wrt m0 centre
            m1_centre = pose_m0_m1 @ np.array([0,0,0,1])
    

            m0_centre = m0_centre[0:2]
            m1_centre = m1_centre[0:2]
            # print(m0_centre)
            # print(m1_centre)

            # #locations of corners relative to centre in meteres
            corner_offests_rw = np.array([[-4.5,4.5],[4.5,4.5],[4.5,-4.5],[-4.5,-4.5]])/100

            #locations of corner pixels relative to centre of m0
            m0_corner_rw = corner_offests_rw + m0_centre

            
            m1_corner_rw = corner_offests_rw  + m1_centre

            pixels = np.vstack((m0.corners,m1.corners))
            #print('pixels',pixels)
            real_world = np.vstack(( m0_corner_rw,m1_corner_rw))
            #print('real world',real_world)



            

            H = cv2.findHomography(pixels,real_world)[0]
            centre_h = point_to_homogenous(centre)
            #print(H)
            x,y,_ = (H @ centre_h)
            print(f'x: {x}, y:{y}')
            
            








            



    count+=1  

    cv2.imshow('not lost in translation',image)
    cv2.waitKey(1)
        

            
        

        




 
   