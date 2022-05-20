from PIL import Image
import numpy as np
class Replacer:

  @staticmethod
  def custom_background(background_file, foreground):

    final_foreground = Image.fromarray(foreground)
    background = Image.open(background_file)
    r, g, b = background.split()
    background = Image.merge("RGB", (b, g, r))

    width = background.size[0]
    height = background.size[1]

    aspect = float(width) / float(height)

    ideal_width = final_foreground.size[0]
    ideal_height = final_foreground.size[1]

    ideal_aspect = float(ideal_width) / float(ideal_height)

    if aspect > ideal_aspect:
      # Then crop the left and right edges:
      new_width = int(ideal_aspect * height)
      offset = (width - new_width) / 2
      resize = (offset, 0, width - offset, height)
    else:
      # ... crop the top and bottom:
      new_height = int(width / ideal_aspect)
      offset = (height - new_height) / 2
      resize = (0, offset, width, height - offset)

    background = background.crop(resize).resize((ideal_width, ideal_height), Image.ANTIALIAS)


    x = (background.size[0]-final_foreground.size[0])/2
    y = (background.size[1]-final_foreground.size[1])/2
    box = (x, y, final_foreground.size[0] + x, final_foreground.size[1] + y)
    crop = background.crop(box)
    final_image = crop.copy()
    # put the foreground in the centre of the background
    paste_box = (0, final_image.size[1] - final_foreground.size[1], final_image.size[0], final_image.size[1])
    final_image.paste(final_foreground, paste_box, mask=final_foreground)

    return np.array(final_image)