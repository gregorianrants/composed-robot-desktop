import cv2

count = 0

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
            #cv2.drawContours(frame,contours,0,(255,0,0),3)
            contour = contours[0]
            x,y,w,h= cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            centre = (x+(w/2),y+(h/2))

        object_of_interest = cv2.bitwise_and(frame,frame,mask=my_mask)
        cv2.imshow('ooi',object_of_interest)
        count+=1
        return centre
