#Phys 201 lab 1 falling ball
#By Pascal R. Jardin
import cv2
import numpy as np
import math

cap = cv2.VideoCapture("/Users/pascaljardin/Desktop/lab1 fall/FALL8.m4v")#enter location of video

# take first frame of the video
b, firstFrame= cap.read()

drawing = False # true if mouse is pressed
cursorX,cursorY = -1,-1

lineStart = (0,0)
lineEnd = (0,0)
recStart = (0,0)
recEnd = (0,0)
drawing = False

drawRes = 0.5

current = "L"

def nothing(x):
    pass

def colorFilter(coleredImage):
    blurred = cv2.GaussianBlur(coleredImage, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global cursorX,cursorY
    cursorX = x
    cursorY = y

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

firstFrame = cv2.resize(firstFrame, (0,0), fx=drawRes, fy=drawRes)
frame_height, frame_width = firstFrame.shape[:2]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('/Users/pascaljardin/Desktop/output.mp4', fourcc, 20.0, (frame_width, frame_height))


cv2.namedWindow('track')


greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

cv2.createTrackbar('minH1','track',greenLower[0],255,nothing)
cv2.createTrackbar('minS2','track',greenLower[1],255,nothing)
cv2.createTrackbar('minV3','track',greenLower[2],255,nothing)
cv2.createTrackbar('maxH4','track',greenUpper[0],255,nothing)
cv2.createTrackbar('maxS5','track',greenUpper[1],255,nothing)
cv2.createTrackbar('maxV6','track',greenUpper[2],255,nothing)

cv2.imshow('filter',colorFilter(firstFrame.copy()))

#----------------first part of program to find hsv range and draw line and box
while(1):
    minH = cv2.getTrackbarPos('minH1','track')
    minS = cv2.getTrackbarPos('minS2','track')
    minV = cv2.getTrackbarPos('minV3','track')
    maxH = cv2.getTrackbarPos('maxH4','track')
    maxS = cv2.getTrackbarPos('maxS5','track')
    maxV = cv2.getTrackbarPos('maxV6','track')
    
    greenLower = (minH, minS, minV)
    greenUpper = (maxH, maxS, maxV)
    cv2.imshow('filter',colorFilter(firstFrame.copy()))
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break


#----------------secound part of program, draw line and box
while(1):
    img = firstFrame.copy()
    cv2.rectangle(img,recStart,recEnd,(0,255,0),2)
    cv2.line(img,lineStart,lineEnd,(255,0,0),5)
    cv2.circle(img,(cursorX,cursorY), 3, (0,0,255), -1)
    cv2.imshow('image',img)
    out.write(img)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('1'):
        lineStart = (cursorX,cursorY)
    elif k == ord('2'):
        lineEnd = (cursorX,cursorY)
    elif k == ord('3'):
        recStart = (cursorX,cursorY)
    elif k == ord('4'):
        recEnd = (cursorX,cursorY)
        #crop = firstFrame[recStart[1]:recEnd[1], recStart[0]:recEnd[0]]
    elif k == 27:
        break


changeOfLine = math.sqrt( math.pow((lineEnd[0] - lineStart[0]),2) + math.pow( (lineEnd[1] - lineStart[1]),2) )

length = 1#m

#-------------very important!--------
rate = 240 #input("what is the frame rate of the video? ")
#-----------------------


ret, frame = cap.read()
frame = cv2.resize(frame, (0,0), fx=drawRes, fy=drawRes)
frame = colorFilter(frame)

# setup initial location of window
# r,h,c,w - region of image
r,h,c,w = recStart[1],recEnd[1] - recStart[1],recStart[0], recEnd[0] - recStart[0]
track_window = (int(c),int(r),int(w),int(h))

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

pointEveryFrame = 10
points = []

font                   = cv2.FONT_HERSHEY_SIMPLEX

curentFrameCount = 0

original = np.zeros((512,512,3), np.uint8)

#----------------third part of program, Calculate gravity
while(1):
    ret ,frame = cap.read()
    
    if ret == True:
        frame = cv2.resize(frame, (0,0), fx=drawRes, fy=drawRes)
        original = frame
        frame = colorFilter(frame)
        

        
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(frame, track_window, term_crit)
        
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(original, (x,y), (x+w,y+h), 255,2)
        xc, yc = (x+w/2,y+h/2)
        
        cv2.line(original,lineStart,lineEnd,(255,0,0),5)
        cv2.putText(original," = 1m ",
            ( int((lineStart[0]+lineEnd[0])/2) ,int((lineStart[1]+lineEnd[1])/2)),
            font,
            0.5,
            (255,255,255),
            1)
        
        for p in points:
            cv2.circle(original,(p["x"],p["y"]), 4, (255,0,0), -1)
            cv2.putText(original,"t = " + str(p["t"]) + "s",
                        (p["x"]+10,p["y"]),
                        font,
                        0.5,
                        (255,255,255),
                        1)

        cv2.imshow('image',original)
        out.write(original)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

        if (curentFrameCount % pointEveryFrame == 0):
            points.append( {"x": int(xc),
                          "y": int(yc),
                          "t": round(curentFrameCount / rate,2)} )

        curentFrameCount += 1
    else:
        cv2.imwrite('/Users/pascaljardin/Desktop/BallFall.png',original)
        break


l = len(points)
if  l > 1:
    prev = points[0]
    cur = points[l-1]
    changeOfX = (cur["x"] - prev["x"])/changeOfLine
    changeOfY = (cur["y"] - prev["y"])/changeOfLine
        
    changeOfH = math.sqrt( math.pow(changeOfX,2) + math.pow(changeOfY,2) )
            
    #I use the  Pythagorean Theorem to merge the change of x and the change of y
    # So H is the combined axis of x and y
    # I know this works since i used this method in a AR drawing app
    # This is need incase the recourding device is not streight relitive to gravity
    changeOfT = cur["t"] - prev["t"]

    """
        a = -g m/s^2
        v = -gt
        y = h -gt^2 /2
        
        find g:
        h = gt^2 / 2
        h*2 = gt^2
        h*2/t^2 = g
        
    """

    g = changeOfH*2/math.pow(changeOfT, 2)
    print("g = ",g)


    for i in range(0, len(points)):

        print("t", points[i]["t"]," x:",points[i]["x"]/changeOfLine," y:",points[i]["y"]/changeOfLine)

#this way i can coppy the values and past them in the code, so i dont have to find hsv range every time!
print ("greenLower: ",greenLower)
print ("greenUpper: ",greenUpper)

out.release()
cv2.destroyAllWindows()
cap.release()





