import cv2
import numpy as np
 
def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin
 
def match_fingerprints(img1_path, img2_path):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)
 
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
 
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None  # Return 0 matches if no descriptors found
 
    # Use Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
 
    # KNN Match
    matches = bf.knnMatch(des1, des2, k=2)
 
    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
 
    # Draw only good matches
    print(len(good_matches))
    return cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

def view_image(image):
    while True:
        cv2.imshow('image', image)
        if cv2.waitKey(1)== ord('q'):
            break

def main():
    view_image(match_fingerprints('figureprint/uia_images/UiA front1.png','figureprint/uia_images/UiA front3.jpg'))



main()