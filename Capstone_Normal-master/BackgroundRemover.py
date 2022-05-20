import  tensorflow as tf
import  numpy as np
import cv2
from segmentation_models.metrics import IOUScore
from segmentation_models.losses import JaccardLoss

class Remover:

    def __init__(self):
        self.model = self.__model_loader__()

    def __model_loader__ (self):
        loss_func = JaccardLoss(per_image=True)
        metric = IOUScore(per_image=True)
        model = tf.keras.models.load_model('model_test', compile=False, custom_objects = {} )
        model.compile(loss=loss_func, optimizer="adam", metrics=[metric])

        return model

    def predict(self, img):
        h, w = img.shape[0:2]

        img_test_resized = cv2.resize(img, (128, 128))
        img_test = np.asarray(img_test_resized) / 255.0
        img_test = img_test[np.newaxis, ...]
        pred_img = self.model.predict(img_test)
        pred_img = np.squeeze(pred_img)

        result = img_test_resized.copy()
        result = cv2.cvtColor(result, cv2.COLOR_RGB2RGBA)

        pred_img_copy = pred_img.copy()*255
        #pred_img_copy[pred_img_copy < 0.5] = 0
        #pred_img_copy[pred_img_copy >= 0.5] = 255  # binarising

        #result[:, :, 3] = pred_img_copy  # adding mask in alpha channel
        # resizing back to original size
        image_height = 128
        image_width = 128
        number_of_color_channels = 3
        color = (0, 0, 0)
        pixel_array = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
        pixel_array[:,:,2] = pred_img_copy
        test = cv2.cvtColor(pixel_array, cv2.COLOR_RGB2GRAY)
        # Search for edges in the image with cv2.Canny().


        # Search for contours in the edged image with cv2.findContour().
        contours, hierarchy = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Filter out contours that are not in your interest by applying size criterion.
        for cnt in contours:
            size = cv2.contourArea(cnt)
            if size > 100:
                cv2.drawContours(pred_img_copy, [cnt], 0, (255, 255, 255 ), 2)
                cv2.drawContours(result, [cnt], 0, (255, 255 , 255), 2)

        result[:, :, 3] = pred_img_copy
        result = cv2.resize(result, (w, h))
        return result