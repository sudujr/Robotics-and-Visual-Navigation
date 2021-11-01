import cv2
cap = cv2.VideoCapture('/home/sudharshan/Documents/Robotics-and-Visual-Navigation/Contrast Limited Adaptive Histogram Equalization/Input/VideoInput/ORIGINAL VIDEO - Night Time image enhancement using CLAHE.mp4')
count = 0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('/home/sudharshan/Documents/Robotics-and-Visual-Navigation/Contrast Limited Adaptive Histogram Equalization/Media/enhancedVideo.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25.0, (frame_width,frame_height))
while cap.isOpened():
    ret,frame = cap.read()
    
    blurredFrame = cv2.GaussianBlur(frame,(9,9),0)
    img2hsv = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)
    v = img2hsv[:,:,2]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16,16))
    cl1 = clahe.apply(v)    
    img2hsv[:,:,2] = cl1
    enhanced_frame = cv2.cvtColor(img2hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('Blurred Image', enhanced_frame)
    out.write(enhanced_frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # destroy all opened windows