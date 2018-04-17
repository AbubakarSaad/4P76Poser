import urllib.request
import cv2
import numpy as np


def getImage():
    url = 'http://192.168.0.3:8080/shot.jpg'
    with urllib.request.urlopen(url) as response:
        imgResp = response.read()
    
    imgNp = np.array(bytearray(imgResp), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    cv2.imwrite('cap.jpg', img)
    c = cv2.waitKey(0)






