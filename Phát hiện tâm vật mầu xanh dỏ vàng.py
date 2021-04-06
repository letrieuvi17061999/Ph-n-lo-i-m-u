import cv2
import numpy as np
import imutils


cap = cv2.VideoCapture(0)


while True:
    _,frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray_frame, 100, 0.1, 20)
    
    if corners is not None:

        corners = np.int0(corners)

        for corner in corners:

            set = corner.ravel()

            x,y = set

            cv2.circle(frame, (x,y), 5, (0,0,255), -1)
        
    # print("tọa độ x {} tọa độ y {}" .format (x,y))
    lower_blue   = np.array([90,60,0])
    upper_blue   = np.array([121,255,255])

    lower_yellow = np.array([25,70,120])
    upper_yellow = np.array([30,225,255])

    lower_green = np.array([40,70,80])
    upper_green = np.array([70,225,255])

    lower_red   = np.array([0,50,120])
    upper_red   = np.array([10,255,255])

    mask_blue   = cv2.inRange (hsv,lower_blue,upper_blue)
    mask_yellow = cv2.inRange (hsv,lower_yellow,upper_yellow)
    mask_green  = cv2.inRange (hsv,lower_green,upper_green)
    mask_red    = cv2.inRange (hsv,lower_red,upper_red)

    cnts_blue   = cv2.findContours(mask_blue, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_blue   = imutils.grab_contours(cnts_blue)

    cnts_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_yellow = imutils.grab_contours(cnts_yellow)

    cnts_green = cv2.findContours(mask_green, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_green = imutils.grab_contours(cnts_green)

    cnts_red    = cv2.findContours(mask_red, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts_red    = imutils.grab_contours(cnts_red)

    for c in cnts_blue:
        area_blue = cv2.contourArea(c) 
        if area_blue > 5000:       

            cv2.drawContours(frame, [c], -1, (0,255,0), 3) 

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"]) 
            cy = int(M["m01"] / M["m00"]) 

            cv2.circle(frame, (cx, cy), 7 ,(225,255,255),-1)
            cv2.putText(frame, "Blue", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,255, 0), 3)           

            # print ("Center blue at x {} và y{}" .format(cx,cy))

    for c in cnts_yellow:
        area_yellow = cv2.contourArea(c) 
        if area_yellow > 5000:       

            cv2.drawContours(frame, [c], -1, (0,255,0), 3) 

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"]) 
            cy = int(M["m01"] / M["m00"]) 

            cv2.circle(frame, (cx, cy), 7 ,(225,255,255),-1)
            cv2.putText(frame, "Yellow", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)           

            # print ("Center yellow at x {} và y{}" .format(cx,cy))

    for c in cnts_green:
        area_green = cv2.contourArea(c) 
        if area_green > 5000:       

            cv2.drawContours(frame, [c], -1, (0,255,0), 3) 

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"]) 
            cy = int(M["m01"] / M["m00"]) 

            cv2.circle(frame, (cx, cy), 7 ,(225,255,255),-1)
            cv2.putText(frame, "Green", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)           

            # print ("Center green at x {} và y{}" .format(cx,cy))
    
    for c in cnts_red:
        area_red = cv2.contourArea(c) 
        if area_red > 5000:       

            cv2.drawContours(frame, [c], -1, (0,255,0), 3) 

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"]) 
            cy = int(M["m01"] / M["m00"]) 

            cv2.circle(frame, (cx, cy), 7 ,(225,255,255),-1)
            cv2.putText(frame, "Red", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 3)           

            # print ("Center red at x {} và y{}" .format(cx,cy))

    cv2.imshow("frame", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
