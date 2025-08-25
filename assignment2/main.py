import cv2
import numpy as np

#pad the image with reflection for specified amount of pixels
def padding(image, amount: int):
    image = cv2.copyMakeBorder(image, amount, amount, 
                               amount, amount, cv2.BORDER_REFLECT)
    return image

#crop the image with specified amount of pixels on each side
def crop(image, x0: int, x1: int, y0: int, y1: int):
    dims = image.shape
    return image[x0:dims[0]-x1 , y0:dims[1]-y1]

#resize image to chosen width and height
def resize(image, width: int, height: int):
    return cv2.resize(image,(width, height))

#copy pixel values to empty array, and return it
def copy(image, emptyArray: np.ndarray):
    emptyArray+=image
    return emptyArray

#returns grayscale version of image
def grayscale(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#converts image to HSV
def rgbToHsv(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#shifts color by specified amount. 
#wraps around if value goes over 255 or under 0
def hue_shifted(img, amount):
    shifted = img+np.ones(img.shape,dtype=np.uint8)*amount
    return shifted

#blurs the image by smoothing out the pixel values
def smoothing(image):
    return cv2.GaussianBlur(image,[15,15],cv2.BORDER_DEFAULT)

#rotates the image in 90 degree intervals. 
def rotation(img: np.ndarray,degrees):
    return np.rot90(img,np.floor(degrees/90))


#show image until "q" is pressed
def showImg(image,title: str= "image"):
    while True:
        cv2.imshow(title, image)
        if cv2.waitKey(1)== ord('q'):
            break
    cv2.destroyAllWindows()




def main():
    #get the image
    img = cv2.imread("assignment2/lena-2.png")

    #1
    imgPadded = padding(img, 100)
    showImg(imgPadded,"padded image with reflection")

    #2
    imgCropped = crop(img,80,120,80,120)
    showImg(imgCropped, "image cropped to only include face")

    #3
    imgResized = resize(img,200,200)
    showImg(imgResized, "image resized to 200x200 pixels")

    #4
    copyImage = np.zeros(img.shape,dtype=np.uint8)
    copy(img,copyImage)
    showImg(copyImage, "copied image")

    #5
    imgGray = grayscale(img)
    showImg(imgGray,"grayscale image")

    #6
    imgHSV = rgbToHsv(img)
    showImg(imgHSV,"HSV image")

    #7
    shiftedImg = hue_shifted(img,50)
    showImg(shiftedImg,"image shifted by 50")

    #8
    smoothedImage = smoothing(img)
    showImg(smoothedImage,"blurred image")

    #9
    rotated90Img = rotation(img,90)
    rotated180Img = rotation(img,180)
    showImg(rotated90Img,"image rotated 90 degs")
    showImg(rotated180Img,"image rotated 180 degs")


main()