import cv2
import os
import numpy as np

paquetes1 = 'E:/utp/2023/semestre 1/Proyecto JIC/data'
listaPersonas = os.listdir(paquetes1)
print('Lista de Personas: ',listaPersonas)

labels = []
facedata = []
label = 0


for nameDir in listaPersonas:
    paquetes = paquetes1 + '/' + nameDir
    print('Leyendo imagenes')

    for fileName in os.listdir(paquetes):
        print('Rostros: ', nameDir+'/'+fileName)
        labels.append(label)

        facedata.append(cv2.imread(paquetes+'/'+fileName,0))
        imagen = cv2.imread(paquetes+'/'+fileName,0)

        cv2.imshow('imagen',imagen)
        cv2.waitKey(10)
    label = label +1 
cv2.destroyAllWindows()

print('labels=  ',labels)
print('Numero de etiquetas:  ',np.count_nonzero(np.array(labels)==0))
print('Numero de etiquetas:  ',np.count_nonzero(np.array(labels)==1))

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print('Entranando')
face_recognizer.train(facedata, np.array(labels))

face_recognizer.write('ModeloFaceRecogniserData2023.xml')
print('Guardando Modelo')

