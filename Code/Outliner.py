import numpy as np
import cv2
class Outline:
    @staticmethod
    def MakeOutlline(img,mask):

        image_height , image_width,_ = img.shape
        mask = cv2.resize(mask,(image_width,image_height))
        number_of_color_channels = 3
        color = (0, 0, 0)
        pixel_array = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
        pixel_array[:, :, 2] = mask
        test = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)
        # Search for edges in the image with cv2.Canny().

        # Search for contours in the edged image with cv2.findContour().
        contours, hierarchy = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Filter out contours that are not in your interest by applying size criterion.
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size > 100:
                cv2.drawContours(mask, [cnt], 0, (255, 255, 255), 2)
                cv2.drawContours(img, [cnt], 0, (255, 255, 255), 2)
        img[:,:,3] = mask
        return img