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
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)

        pred_img_copy = pred_img.copy()
        pred_img_copy[pred_img_copy < 0.5] = 0
        pred_img_copy[pred_img_copy >= 0.5] = 255  # binarising

        result[:, :, 3] = pred_img_copy  # adding mask in alpha channel
        result = cv2.resize(result, (h, w))  # resizing back to original size

        return result

