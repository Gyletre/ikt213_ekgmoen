import numpy as np
import cv2

class FingerPrintDetector:
    def __init__(self,fast: bool):
        if(fast):
            self.model=cv2.ORB_create(nfeatures=1000)
        else:
            self.model=cv2.SIFT_create()
    def read(self,image):
        keypoints, descriptions = self.model.detectAndCompute(image, None)

        return cv2.drawKeypoints(image,keypoints, None, color=(0,255,0))
    
    def view_image(self,image):
        while True:
            cv2.imshow('image', image)
            if cv2.waitKey(1)== ord('q'):
                break
        
def main():
    image = cv2.imread('figureprint/uia_images/UiA front1.png')
    fpd = FingerPrintDetector(False)
    fpd.view_image(fpd.read(image))

main()

