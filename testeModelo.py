import numpy as np
import cv2
from keras.models import load_model
import mediapipe as mp

face = mp.solutions.face_detection
Face = face.FaceDetection()
mpDwaw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('Videos/vd (1).mp4')

model = load_model('modelo.h5')

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getCalssName(classNo):
    if classNo == 0:
        return 'Nervoso'
    elif classNo == 1:
        return 'Nojo'
    elif classNo == 2:
        return 'Medo'
    elif classNo == 3:
        return 'Felicidade'
    elif classNo == 4:
        return 'Neutro'
    elif classNo == 5:
        return 'Triste'
    elif classNo == 6:
        return 'Surpreso'

while True:
    success, imgOrignal = cap.read()

    imgRGB = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2RGB)
    results = Face.process(imgRGB)
    facesPoints = results.detections
    hO, wO, _ = imgRGB.shape
    if facesPoints:
        for id, detection in enumerate(facesPoints):
            #mpDwaw.draw_detection(img, detection)
            bbox = detection.location_data.relative_bounding_box
            x,y,w,h = int(bbox.xmin*wO),int(bbox.ymin*hO),int(bbox.width*wO),int(bbox.height*hO)
            imagemFace = imgOrignal[y-30:y+h,x:x+w]
            img = np.asarray(imagemFace)
            img = cv2.resize(img, (48, 48))
            img = preprocessing(img)
            img = img.reshape(1, 48, 48, 1)

            predictions = model.predict(img)
            indexVal = np.argmax(predictions)
            probabilityValue = np.amax(predictions)
            print(indexVal,probabilityValue)
            if probabilityValue >0.20:
                cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0,255,0), 3)

                cv2.putText(imgOrignal, str(getCalssName(indexVal)), (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 3,
                            (0,255,0), 8, cv2.LINE_AA)

                cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0), 2,
                            cv2.LINE_AA)

    cv2.imshow("Result", cv2.resize(imgOrignal,(0,0),None,0.40,0.40))
    #cv2.imshow("Face", imagemFace)
    cv2.waitKey(15)

