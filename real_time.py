import cv2, sys

def realTimeCapture():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0
    ret, frame = cam.read()
    cv2.imshow("test", frame)

    img_name = sys.path[0] + r"/../images/LiveTest/cap.png"
    cv2.imwrite(img_name, frame)
    print("written!")

    cam.release()

    cv2.destroyAllWindows()