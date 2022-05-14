from PIL import Image
import numpy as np
class Replacer:

  @staticmethod
  def custom_background(background_file, foreground):

    final_foreground = Image.fromarray(foreground)
    background = Image.open(background_file)
    x = (background.size[0]-final_foreground.size[0])/2
    y = (background.size[1]-final_foreground.size[1])/2
    box = (x, y, final_foreground.size[0] + x, final_foreground.size[1] + y)
    crop = background.crop(box)
    final_image = crop.copy()
    # put the foreground in the centre of the background
    paste_box = (0, final_image.size[1] - final_foreground.size[1], final_image.size[0], final_image.size[1])
    final_image.paste(final_foreground, paste_box, mask=final_foreground)

    return np.array(final_image)