import cv2 #opencv
 #para abrir y leer URL
import numpy as np
import time

url = 'http://192.168.0.15/cam-hi.jpg'
#url = 'http://192.168.1.6/'
winName = 'ESP32 CAMERA 001'
cv2.namedWindow(winName,cv2.WINDOW_AUTOSIZE)
#scale_percent = 80 # percent of original size    #para procesamiento de imagen
while(1):
    imgNp = np.array(bytearray('http://192.168.0.15/cam-hi.jpg'),dtype=np.uint8)
    img = cv2.imdecode (imgNp,-1) #decodificamos
    cv2.imshow(winName,img) # mostramos la imagen
    #esperamos a que se presione ESC para terminr el programa
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break
cv2.destroyAllWindows()
