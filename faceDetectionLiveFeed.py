import cv2

# loading the haar cascade (unmodified version from openCV3.2)
cascPath = "haarcascade_frontalface_default.xml"
faceHaarCascade = cv2.CascadeClassifier(cascPath)

# Init Video capture from the very first found camera (system default)
videoCapture = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame and apply the haar cascade
    ret, videoFrameColor = videoCapture.read()
    videoFrameGray = cv2.cvtColor(videoFrameColor, cv2.COLOR_BGR2GRAY)
    detectedFaces = faceHaarCascade.detectMultiScale(videoFrameGray, scaleFactor = 1.2, minNeighbors = 3, minSize=(10,10))

    # Draw the rectangles around the face(s)
    # (255,255,0) -> turquoise color and the '3' right after is the thickness
    #  openCV's notation for color is BGR not RGB.
    for (x,y,width,height) in detectedFaces:
        cv2.rectangle(videoFrameColor, (x,y), (x+width, y+height), (255, 255, 0), 3)

    # On every single iteration check if the user pushes 'q' on the keyboard.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame (continuous update)
    cv2.imshow('Live Feed',videoFrameColor)

# When everything done, release the capture
videoCapture.release()
