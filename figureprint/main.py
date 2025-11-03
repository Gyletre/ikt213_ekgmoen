import numpy as np
import cv2
import time

class FingerPrintDetector:
    def __init__(self,useOrb: bool,useFlann,nfeats, preprocessing = False):
        if(useOrb):
            self.model=cv2.ORB_create(nfeatures=nfeats)
        else:
            self.model=cv2.SIFT_create(nfeatures=nfeats)
        if useFlann:
            index_params = dict(algorithm=1, trees=5)  # KD-tree
            search_params = dict(checks=50)  # Number of checks for nearest neighbors
            self.processing = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self.processing = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.preprocessing = preprocessing
    def read(self,image):
        keypoints, descriptions = self.model.detectAndCompute(image, None)

        return cv2.drawKeypoints(image,keypoints, None, color=(0,255,0))
    
    def view_image(self,image):
        while True:
            cv2.imshow('image', image)
            if cv2.waitKey(1)== ord('q'):
                break
    def preprocess_fingerprint(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return img_bin
    def compare_images(self, img1, img2,skipgraphics = False):
        #_, img_bin1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #_, img_bin2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if self.preprocessing:
            img1 = self.preprocess_fingerprint(img1)
            img2 = self.preprocess_fingerprint(img2)

        kp1, des1 = self.model.detectAndCompute(img1, None)
        kp2, des2 = self.model.detectAndCompute(img2, None)

        if(des1 is None or des2 is None):
            return
        try:
            matches = self.processing.knnMatch(des1, des2, k=2)

            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
            match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            print(len(good_matches),end=" ")
            if(skipgraphics):
                return len(good_matches)
            self.view_image(match_img)
            return len(good_matches)
        except:
            print("failed!",end=" ")
            return 0
        
def main():
    image = cv2.imread('figureprint/uia_images/UiA front1.png',1)
    image2 = cv2.imread('figureprint/uia_images/UiA front3.jpg',1)
    feats = 10000
    models = {}
    preprocessing = True
    models["siftFlann"] =   FingerPrintDetector(useOrb=False,useFlann=True,nfeats=feats,preprocessing=preprocessing)
    models["orbBf"]     =   FingerPrintDetector(useOrb=True,useFlann=False,nfeats=feats,preprocessing=preprocessing)
    
    times = []
    avgs = []
    for name, fpd in models.items():
        print("\n",name,sep="")
        matches = []
        timestart = time.time()
        for i in range(1):
            matches.append(fpd.compare_images(image,image2,False))
        times.append(time.time()-timestart)
        avgs.append(sum(matches)/len(matches))
    print("""
Times:
{:<10}:  time {:.3f}, avgmatches {}
{:<10}:  time {:.3f}, avgmatches {} """.format(*[k for t in zip(models.keys(),times,avgs) for k in t]))

main()

