import cv2



def unwarp(image,calibration_matrix):
    camera_matrix, camera_distortion = calibration_matrix
    h,w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, camera_distortion, (w,h), 1, (w,h))
    image = cv2.undistort(image, camera_matrix, camera_distortion, None, newcameramtx)
    x, y, w, h = roi
    image = image[y:y+h, x:x+w]
    return image