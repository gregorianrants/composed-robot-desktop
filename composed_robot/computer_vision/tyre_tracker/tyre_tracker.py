import cv2
import numpy as np
from pathlib import Path

count = 0


def inRange_wrapped(frame_hsv,lower_bound,upper_bound):
    if (lower_bound[0]<upper_bound[0]):
        my_mask = cv2.inRange(frame_hsv,lower_bound,upper_bound)
    else:
        lower_bound_1 = lower_bound.copy()
        upper_bound_1 = upper_bound.copy()
        upper_bound_1[0] = 179
        lower_bound_2 = lower_bound.copy()
        upper_bound_2 = upper_bound.copy()
        lower_bound_2[0] = 0
        mask_1 = cv2.inRange(frame_hsv,lower_bound_1,upper_bound_1)
        mask_2 = cv2.inRange(frame_hsv,lower_bound_2,upper_bound_2)
        my_mask = mask_1 | mask_2
    return my_mask
    

# def track_tyre(frame,lower_bound,upper_bound):
#         global count
#         frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#         centre = False
#         height,width,_ = frame.shape
#         centre_pixel = frame_hsv[height//2,width//2]
#         # if count%15==0:   
#         #     print('centre pixel hsv values')
#         #     print(centre_pixel)
        
#         my_mask = cv2.inRange(frame_hsv,lower_bound,upper_bound)

#         contours,junk = cv2.findContours(my_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#         if len(contours)>0:
#             contours = sorted(contours, key=lambda x:cv2.contourArea(x),reverse=True)
#             #cv2.drawContours(frame,contours,0,(255,0,0),3)
#             contour = contours[0]
#             x,y,w,h= cv2.boundingRect(contour)
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
#             centre = (x+(w/2),y+(h/2))
#             if count%15==0:
#                 print(cv2.contourArea(contour))

#         object_of_interest = cv2.bitwise_and(frame,frame,mask=my_mask)
#         cv2.imshow('ooi',object_of_interest)
#         count+=1
        
#         return centre
    

def track_tyre(frame,lower_bound,upper_bound):
        global count
        frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        centre = False
        height,width,_ = frame.shape
        centre_pixel = frame_hsv[height//2,width//2]
        # if count%15==0:   
        #     print('centre pixel hsv values')
        #     print(centre_pixel)
        
        my_mask = cv2.inRange(frame_hsv,lower_bound,upper_bound)
        contours,junk = cv2.findContours(my_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>0:
            contours = sorted(contours, key=lambda x:cv2.contourArea(x),reverse=True)
            contour_sizes = [cv2.contourArea(x) for x in contours]
            #np.save(Path(__file__).parent/'contour_sizes',np.array(contour_sizes))
            contour_sizes = np.array(contour_sizes)
            #cv2.drawContours(frame,contours,0,(255,0,0),3)
            contour_has_correct_size = (contour_sizes>50) & (contour_sizes<3000)
            if contour_has_correct_size.sum()==0:
                centre = False
                pass
            best_contour_index = np.argmax(contour_has_correct_size)
            best_contour = contours[best_contour_index]
            x,y,w,h= cv2.boundingRect(best_contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            centre = (x+(w/2),y+(h/2))
            if count%15==0:
                print(cv2.contourArea(best_contour))
                
        
        object_of_interest = cv2.bitwise_and(frame,frame,mask=my_mask)
        cv2.imshow('ooi',object_of_interest)
        cv2.imshow('camera',frame)
        count+=1
        return centre
    


