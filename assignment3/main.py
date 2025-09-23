import cv2

def sobel_edge_detection(image):
    blimage = cv2.GaussianBlur(image,[3,3],SigmaX=0)
    sobel_img =cv2.Sobel(src=blimage,ddepth=cv2.CV_64F,dx=1,dy=1,ksize=1)
    cv2.imshow(sobel_img)
    cv2.waitkey(0)
    pass
def canny_edge_detection(image):
    pass

sobel_edge_detection(cv2.imread("assignment3/lambo.png"))
