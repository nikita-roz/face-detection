import cv2

imagePath = "images/new-museum-wafaces-banner.jpg"
cascPath = "haarcascade_frontalface_default.xml"

#Load the Pics
faceImageColor = cv2.imread(imagePath)
faceImageGray = cv2.cvtColor(faceImageColor, cv2.COLOR_BGR2GRAY)

# loading the haar cascade (unmodified version from openCV3.2)
faceHaarCascade = cv2.CascadeClassifier(cascPath)
detectedFaces = faceHaarCascade.detectMultiScale(faceImageGray, scaleFactor = 1.2, minNeighbors = 3, minSize=(10,10))
print ("# of detected Faces: {0}".format(len(detectedFaces)))

# Draw the rectangles around the faces. The addtl. if statement is just a noise reducer if necessary
# (255,255,0) -> turquoise color and the '3' right after is the thickness
#  openCV's notation for color is BGR not RGB. 
for (x,y,width,height) in detectedFaces:
#    if (y < 1000):
        cv2.rectangle(faceImageColor, (x,y), (x+width, y+height), (255, 255, 0), 3)

# Print data of rectangles in terminal if necessary
#for (x,y,width,height) in detectedFaces:
#    print ("x {0}, y {1}".format(x, y))
faceImageColorResized = cv2.resize(faceImageColor, (0,0), fx=0.8, fy=0.8)
cv2.imshow("Random Faces", faceImageColorResized)
cv2.waitKey(0)
cv2.destroyAllWindows()
