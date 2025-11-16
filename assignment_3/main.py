import cv2
import numpy as np
import os.path as path

def sobel_edge_detection(image):
    blimage = cv2.GaussianBlur(image,[3,3],sigmaX=0)
    sobel_img =cv2.Sobel(src=blimage,ddepth=cv2.CV_64F,dx=1,dy=1,ksize=1)
    cv2.imshow("image",sobel_img)
    cv2.waitKey(0)
    cv2.imwrite("assignment3/images/sobel.png",sobel_img)
    
    
def canny_edge_detection(image,threshold1,threshold2):
    blimage = cv2.GaussianBlur(image,[3,3],sigmaX=0)
    canny_img = cv2.Canny(blimage,threshold1=threshold1,threshold2=threshold2)
    cv2.imshow("image",canny_img)
    cv2.waitKey(0)
    cv2.imwrite("assignment3/images/canny.png",canny_img)

def template_match(image, template):
    w,h = template.shape[::-1]
    result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(result >= threshold)
    for point in zip(*loc[::-1]):
        cv2.rectangle(image,point,(point[0]+w,point[1]+h),(0,0,0),3)
    cv2.imshow("result",image)
    cv2.waitKey(0)
    cv2.imwrite("assignment3/images/template.png",image)

def resize(image,factor: int, up_or_down:str):
    rows, cols, channels = map(int,image.shape)
    if up_or_down == "up":
        img = cv2.pyrUp(image,dstsize=(factor*cols,factor*rows))
        cv2.imshow("zoom in",image)
        cv2.waitKey(0)
        cv2.imwrite("assignment3/images/zoom_in.png",img)
    elif up_or_down == "down":
        img = cv2.pyrDown(image,dstsize=(cols//factor,rows//factor))
        cv2.imshow("zoom out",image)
        cv2.waitKey(0)
        cv2.imwrite("assignment3/images/zoom_out.png",img)
    else:
        print("up_or_down must be either \"up\" or \"down\"")





image1 = cv2.imread(path.dirname(__file__) +"/lambo.png")
sobel_edge_detection(image1)
canny_edge_detection(image1,50,50)

image2 = cv2.cvtColor(cv2.imread(path.dirname(__file__) +"/shapes-1.png"),cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(cv2.imread(path.dirname(__file__) +"/shapes_template.jpg"),cv2.COLOR_BGR2GRAY)

template_match(image2,template)

resize(image1,2,"up")
resize(image1,2,"down")


