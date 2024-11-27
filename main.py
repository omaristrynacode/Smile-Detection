import cv2
#face detector from google
facePath = "facedetector.xml"
faceCascade = cv2.CascadeClassifier(facePath)

#mouth detector from google
smilePath = "smiledetector.xml"
smileCascade = cv2.CascadeClassifier(smilePath)

#use any picture of a person with a smile 
img = cv2.imread("ladyimg.jpg")  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#first detect the face and return the rectangular frame surrounding the face
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor= 1.1,
    minNeighbors=8,
    minSize=(55, 55),
    flags=cv2.CASCADE_SCALE_IMAGE
)

#draw the face and extract the area where the face is located
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # detecting the smile
    smile = smileCascade.detectMultiScale(
        roi_gray,
        scaleFactor= 1.16,
        minNeighbors=35,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #frame the raised corners of the mouth and label the smile with 'Smile'
    for (x2, y2, w2, h2) in smile:
        cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
        cv2.putText(img,'Smile',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('Smile Detector:', img)

c = cv2.waitKey(0)