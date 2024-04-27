import numpy as np
import cv2
from keras.models import load_model

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
        return 'Raiva'
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

classificador = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    success, imgOrignal = cap.read()

    imgGray = cv2.cvtColor(imgOrignal, cv2.COLOR_BGR2GRAY)
    facesPoints = classificador.detectMultiScale(imgGray, scaleFactor=1.5, minSize=(50, 50))
    hO, wO, _ = imgOrignal.shape

    if len(facesPoints)>=1:
        for (x,y,w,h) in facesPoints:
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

                cv2.putText(imgOrignal, str(getCalssName(indexVal)), (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0,255,0), 8, cv2.LINE_AA)

                cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,255,0), 2,
                            cv2.LINE_AA)

    cv2.imshow("Result",imgOrignal)
    #cv2.imshow("Face", imagemFace)
    cv2.waitKey(15)

