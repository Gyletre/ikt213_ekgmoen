import numpy as np
import cv2

class FingerPrintDetector:
    def __init__(self,fast: bool,nfeats):
        if(fast):
            self.model=cv2.ORB_create(nfeatures=nfeats)
        else:
            self.model=cv2.SIFT_create(nfeatures=nfeats)
    def read(self,image):
        keypoints, descriptions = self.model.detectAndCompute(image, None)

        return cv2.drawKeypoints(image,keypoints, None, color=(0,255,0))
    
    def view_image(self,image):
        while True:
            cv2.imshow('image', image)
            if cv2.waitKey(1)== ord('q'):
                break
    def compare_images(self, img1, img2):
        #_, img_bin1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #_, img_bin2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kp1, des1 = self.model.detectAndCompute(img1, None)
        kp2, des2 = self.model.detectAndCompute(img2, None)

        if(des1 is None or des2 is None):
            return
        index_params = dict(algorithm=1, trees=5) 
        search_params = dict(checks=50)  
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        print(len(good_matches))
        self.view_image(match_img)
        
def main():
    image = cv2.imread('figureprint/uia_images/UiA front1.png',1)
    image2 = cv2.imread('figureprint/uia_images/UiA front3.jpg',1)
    fpd = FingerPrintDetector(False,1000)
    fpd.compare_images(image,image2)



main()

