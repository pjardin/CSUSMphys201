#Phys 201 lab 2 parachute
#By Pascal R. Jardin
#------------------ opencv
import cv2
import numpy as np
import math
#------------------
#how to python to excel https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html?utm_source=youtube&utm_medium=video&utm_campaign=youtube_python_google_sheets
import gspread
from oauth2client.service_account import ServiceAccountCredentials
#------------------


# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/pascaljardin/Desktop/lab2 parachute/client_secret.json', scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
nameOfSpreadsheet = "parachute"
sheet = client.open(nameOfSpreadsheet).sheet1

VideoLocation = "/Users/pascaljardin/Desktop/lab2 parachute/FALL3.m4v"

cap = cv2.VideoCapture(VideoLocation)#enter location of video

# take first frame of the video
b, firstFrame= cap.read()

drawing = False # true if mouse is pressed
cursorX,cursorY = -1,-1

""" fall 4

"""
""" fall 3
    greenLower =  (123, 142, 0)
    greenUpper =  (255, 255, 255)
    lineStart =  (78, 122)
    lineEnd =  (80, 162)
    recStart =  (115, 93)
    recEnd =  (126, 107)
    trackStart =  (61, 102)
    trackEnd =  (82, 122)
"""
greenLower =  (123, 142, 0)
greenUpper =  (255, 255, 255)
lineStart =  (78, 122)
lineEnd =  (80, 162)
recStart =  (115, 93)
recEnd =  (126, 107)
trackStart =  (61, 102)
trackEnd =  (82, 122)

drawing = False

drawRes = 0.5

current = "L"

def nothing(x):
    pass

def colorFilter(coleredImage):
    blurred = cv2.GaussianBlur(coleredImage, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=10)
    return mask

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global cursorX,cursorY
    cursorX = x
    cursorY = y

# Print iterations progress
#https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
        Call in a loop to create terminal progress bar
        @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

firstFrame = cv2.resize(firstFrame, (0,0), fx=drawRes, fy=drawRes)
frame_height, frame_width = firstFrame.shape[:2]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use the lower case
out = cv2.VideoWriter('/Users/pascaljardin/Desktop/output.mp4', fourcc, 20.0, (frame_width, frame_height))


cv2.namedWindow('track')




cv2.createTrackbar('minH1','track',greenLower[0],255,nothing)
cv2.createTrackbar('minS2','track',greenLower[1],255,nothing)
cv2.createTrackbar('minV3','track',greenLower[2],255,nothing)
cv2.createTrackbar('maxH4','track',greenUpper[0],255,nothing)
cv2.createTrackbar('maxS5','track',greenUpper[1],255,nothing)
cv2.createTrackbar('maxV6','track',greenUpper[2],255,nothing)

cv2.imshow('filter',colorFilter(firstFrame.copy()))

#----------------first part of program to find hsv range and draw line and box

while(1):
    
    ret ,frame = cap.read()
    
    if ret == True:
        frame = cv2.resize(frame, (0,0), fx=drawRes, fy=drawRes)

        minH = cv2.getTrackbarPos('minH1','track')
        minS = cv2.getTrackbarPos('minS2','track')
        minV = cv2.getTrackbarPos('minV3','track')
        maxH = cv2.getTrackbarPos('maxH4','track')
        maxS = cv2.getTrackbarPos('maxS5','track')
        maxV = cv2.getTrackbarPos('maxV6','track')
        
        greenLower = (minH, minS, minV)
        greenUpper = (maxH, maxS, maxV)
        cv2.imshow('filter',colorFilter(frame.copy()))
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:
            break
    else :
        cap = cv2.VideoCapture(VideoLocation)

cv2.waitKey(1)
cv2.destroyWindow('filter')
cv2.waitKey(1)
cv2.destroyWindow('track')
cv2.waitKey(1)

#----------------secound part of program, draw line and box
while(1):
    img = firstFrame.copy()
    cv2.rectangle(img,trackStart,trackEnd,(255,255,255),2)
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
    elif k == ord('5'):
        trackStart = (cursorX,cursorY)
    elif k == ord('6'):
        trackEnd = (cursorX,cursorY)
    #crop = firstFrame[recStart[1]:recEnd[1], recStart[0]:recEnd[0]]
    elif k == 27:
        break


changeOfLine = math.sqrt( math.pow((lineEnd[0] - lineStart[0]),2) + math.pow( (lineEnd[1] - lineStart[1]),2) )

length = 1#m

#----------------third part of program, Calculate drag

#-------------very important!--------frame
rate = 30 #input("what is the frame rate of the video? ")
#-----------------------

cap = cv2.VideoCapture(VideoLocation)

ret, frame = cap.read()
frame = cv2.resize(frame, (0,0), fx=drawRes, fy=drawRes)
frame = colorFilter(frame)

# setup initial location of window
# r,h,c,w - region of image
r,h,c,w = recStart[1],recEnd[1] - recStart[1],recStart[0], recEnd[0] - recStart[0]
track_window = (int(c),int(r),int(w),int(h))

rr,rh,rc,rw = trackStart[1],trackEnd[1] - trackStart[1],trackStart[0], trackEnd[0] - trackStart[0]
reference_widow = (int(rc),int(rr),int(rw),int(rh))
rx,ry,rw,rh = reference_widow

originalX, originalY = (int(rx+rw/2),int(ry+rh/2))


print(track_window)
print(reference_widow)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

pointEveryFrame = 10
points = []

font                   = cv2.FONT_HERSHEY_SIMPLEX

curentFrameCount = 0

original = np.zeros((512,512,3), np.uint8)

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
        cv2.rectangle(original, (x,y), (x+w,y+h), 255,5)
        xc, yc = (x+w/2,y+h/2)
        
        # apply meanshift to get the new location
        ret1, reference_widow = cv2.CamShift(frame, reference_widow, term_crit)
        
        # Draw it on image
        rx,ry,rw,rh = reference_widow
        cv2.rectangle(original, (rx,ry), (rx+rw,ry+rh), (255,255,255),2)
        rxc, ryc = (int(rx+rw/2),int(ry+rh/2))
        
        shiftFromX = rxc - originalX
        shiftFromY = ryc - originalY

        newLineStart = (lineStart[0] + shiftFromX,lineStart[1] + shiftFromY)
        newLineEnd = (lineEnd[0] + shiftFromX,lineEnd[1] + shiftFromY)

        cv2.line(original,newLineStart,newLineEnd,(255,0,0),5)
        
        
        cv2.putText(original," = 1m ",
                    ( int((lineStart[0]+lineEnd[0])/2) ,int((lineStart[1]+lineEnd[1])/2)),
                    font,
                    0.5,
                    (255,255,255),
                    1)
            
        for i in range(0, len(points)):
                
            p = points[i]
            if (i % pointEveryFrame == 0):
                cv2.circle(original,(p["x"]+rxc,p["y"]+ryc), 4, (255,0,0), -1)
                cv2.putText(original,"t = " + str(round(p["t"],2)) + "s",
                            (p["x"]+10+rxc,p["y"]+ryc),
                            font,
                            0.5,
                            (255,255,255),
                            1)
                    
        cv2.imshow('image',original)
        out.write(original)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
                
        points.append( {"x": int(xc-rxc),
                        "y": int(yc-ryc),
                        "t": curentFrameCount / rate} )
                
        curentFrameCount += 1

    else:
        cv2.imwrite('/Users/pascaljardin/Desktop/BallFall.png',original)
        break

cv2.waitKey(1)
out.release()
cv2.destroyAllWindows()
cap.release()
cv2.waitKey(1)



#sends data to spreadsheet

print("to spreadsheet ", nameOfSpreadsheet)

l = len(points)
nextCell = 0
if  l > 1:
    
    startAtCellX = 20
    startAtCellY = 7

    for i in range(0, len(points)):
        #if (i % 3 == 0):

        p = points[i]

        sheet.update_cell(startAtCellX + nextCell, startAtCellY, p["t"])
        sheet.update_cell(startAtCellX + nextCell, startAtCellY+1, p["y"]/changeOfLine)

        printProgressBar(i, len(points)-1, prefix = 'Progress:', suffix = 'Complete', length = 50)
        nextCell += 1

printProgressBar(len(points)-1, len(points)-1, prefix = 'Progress:', suffix = 'Complete', length = 50)

#this way i can coppy the values and past them in the code, so i dont have to find hsv range every time!
print ("greenLower = ",greenLower)
print ("greenUpper = ",greenUpper)
print("lineStart = ", lineStart)
print("lineEnd = ", lineEnd)
print("recStart = ", recStart)
print("recEnd = ", recEnd)
print("trackStart = ", trackStart)
print("trackEnd = ", trackEnd)

