import cv2
import os
import numpy as np

dirname = os.path.dirname(__file__)

#1
def harris_corner_detection(referance_image, toFile = True):
    img = referance_image.copy()
    gray = np.float32(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)
    
    img[dst>0.01*dst.max()]=[0,0,255]
    
    if (toFile):
        cv2.imwrite(dirname+"/images/harris_corner_detection.png",img)
    else:
        cv2.imshow("corner detection",img)
        cv2.waitKey(0)
        return max(dst,key=lambda item: item[0])

#2 
# Wanted to use SURF since it was not used in the figureprint task.
# However, because it is patented and marked as nonfree, SIFT was used instead.
def align_imgs_SIFT(image_to_align, referance_image, max_features, good_match_percent):
    if image_to_align is None or referance_image is None: 
        print('Image(s) not found')
        return
    
    # Find keypoints and descriptors
    detector = cv2.SIFT.create(max_features)
    kp1, desc1 = detector.detectAndCompute(image_to_align, None)
    kp2, desc2 = detector.detectAndCompute(referance_image, None)
    
    flann_index_kdtree = 1
    index_parameters = dict(algorithm = flann_index_kdtree, trees = 5)
    search_parameters = dict(checks=50)
    
    matcher = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    knn_matches = matcher.knnMatch(desc1, desc2, 2)
    
    good_matches = []
    for m,n in knn_matches:
        if m.distance < good_match_percent * n.distance:
            good_matches.append(m)
    
    img_matches = np.empty((max(image_to_align.shape[0], referance_image.shape[0]), image_to_align.shape[1]+referance_image.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(image_to_align, 
                    kp1, 
                    referance_image, kp2, 
                    good_matches, 
                    img_matches, 
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv2.imwrite(dirname+"/images/SIFT_aligned_with_min_acc={}%.png".format(good_match_percent),img_matches)

referance_image = cv2.imread(dirname + "/reference_img.png")
align_image = cv2.imread(dirname + "/align_this.jpg")

harris_corner_detection(referance_image) 
harris_corner_detection(align_image,False)


align_imgs_SIFT(align_image,referance_image,100,75)

