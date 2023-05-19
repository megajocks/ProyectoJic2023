import cv2
import imutils
import os




persoName = 'personas'
data = 'E:/utp/2023/semestre 1/Proyecto JIC/data'
paquetes = data + '/' + persoName

if not os.path.exists(paquetes):
    print('Carpeta Creada:   ',paquetes)
    os.makedirs(paquetes)


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

while True:
    
    ret , frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=320)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxframe = frame.copy()

    caras = faceClassif.detectMultiScale(gray, 1.3,5)

    for (x,y,w,h) in caras:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        rostro = auxframe[y:y + h, x:x + w]
        rostro = cv2.resize(rostro,(720,720), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(paquetes+'/rostro_{}.jpg'.format(count), rostro)
        count = count + 1 


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break
cap.release()
cv2.destroyAllWindows()
    

