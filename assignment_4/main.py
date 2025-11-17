import cv2
import os
import numpy as np

dirname = os.path.dirname(__file__)

#1
def harris_corner_detection(reference_image, toFile = True):
    img = reference_image.copy()
    gray = np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    
    img[dst>0.01*dst.max()]=[0,0,255]
    
    if (toFile):
        cv2.imwrite(dirname+"/images/harris_corner_detection.png",img)
    else:
        #cv2.imshow("corner detection",img)
        cv2.waitKey(0)
        return max(dst,key=lambda item: item[0])

#2 
# Wanted to use SURF since it was not used in the figureprint task.
# However, because it is patented and marked as nonfree, I did not use it
# Sift reference code had no implementation of alignment, only matching, so I used ORB

def align_imgs_ORB(image_to_align, reference_image, max_features, good_match_percent):

    align_gray = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    referance_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(max_features)
    kp1, desc1 = orb.detectAndCompute(align_gray, None)
    kp2, desc2 = orb.detectAndCompute(referance_gray, None)
    
    # Create brute force matcher and match with ORB descriptors
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2, None)
    
    # Sort matches by accuracy
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Only keep best matches 
    matchesToKeep = int(len(matches) * good_match_percent)
    matches = matches[:matchesToKeep]

    # Draw matches and save to image
    imMatches = cv2.drawMatches(image_to_align, kp1, reference_image, kp2, matches, None)
    cv2.imwrite(dirname+"/images/matches.jpg", imMatches)
    
    # Create empty matrix and fill with x,y coordinates of matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    # Calculate how to warp, then align image
    M, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    height, width, _ = reference_image.shape
    warped_image = cv2.warpPerspective(image_to_align, M, (width, height))
    
    cv2.imwrite(dirname+"/images/aligned_img.png", warped_image)

reference_image = cv2.imread(dirname + "/reference_img.png")
align_image = cv2.imread(dirname + "/align_this.jpg")

harris_corner_detection(reference_image) 
harris_corner_detection(align_image,False)
cv2.destroyAllWindows()

align_imgs_ORB(align_image,reference_image,10000,0.1)

