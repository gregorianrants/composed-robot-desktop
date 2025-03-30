import numpy as np

def get_calibration_matrix():
    with open('composed_robot/computer_vision/camera_calibration/camera_cal_3.npy','rb') as f:
        camera_matrix = np.load(f)
        camera_distortion = np.load(f)
        
    f_x = camera_matrix[0,0]
    f_y = camera_matrix[1,1]
    o_x = camera_matrix[0,2]
    o_y = camera_matrix[1,2]

    #print(f_x,f_y,o_x,o_y)

    dimx_new = 820
    dimx_old = 1640
    dimy_new = 486
    dimy_old = 1232
    f_x = (dimx_new/dimx_old) * f_x
    f_y = (dimy_new/dimy_old) * f_y
    o_x = (dimx_new/dimx_old) * o_x
    o_y = (dimy_new/dimy_old) * o_y
    camera_matrix[0,0] = f_x
    camera_matrix[1,1] = f_y
    camera_matrix[0,2] = o_x
    camera_matrix[1,2] = o_y 
    return camera_matrix,camera_distortion