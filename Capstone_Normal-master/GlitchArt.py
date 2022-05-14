import pixelsort
import numpy as np
from PIL import Image



class GlitchEffect:
    @staticmethod
    def Glitch(image):
        image = Image.fromarray(image)
        return np.array(pixelsort.pixelsort(image))