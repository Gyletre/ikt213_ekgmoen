import cv2
import numpy as np
def print_image_information(image):
    img = cv2.imread(image, -1)
    print("image information:\n")
    shape = img.shape
    print("shape:\n" 
    f" width: {shape[0]}\n"
    f" height: {shape[1]}\n"
    f" channels: {shape[2]}")

    print("\nsize:",img.size)
    print("data type:", image[-4:])

def save_video_specifications(cam):
    cameraInfo = []
    cameraInfo.append(f"fps: {cam.get(cv2.CAP_PROP_FPS)}")
    cameraInfo.append(f"width: {cam.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    cameraInfo.append(f"height: {cam.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    np.savetxt("assignment1/solutions/camera_outputs.txt", cameraInfo,fmt="%s")

def main():
    cam = cv2.VideoCapture(0)
    print_image_information("assignment1/lena-1.png")
    save_video_specifications(cam)


main()