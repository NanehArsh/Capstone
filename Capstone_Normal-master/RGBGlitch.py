import numpy as np
from PIL import Image, ImageOps

class RGBGlitch:

  @staticmethod
  def rgbglitch(background_file, foreground):

    final_foreground = Image.fromarray(foreground)
    background = Image.open(background_file)
    r, g, b = background.split()
    background = Image.merge("RGB", (b, g, r))

    r, g, b, a = final_foreground.split()
    r1 = resize_with_padding(r, (int(1.05 * r.size[0]), r.size[1]))
    r2 = r1.crop((0, 0, r.size[0], r.size[1]))

    g1 = resize_with_padding(g, (g.size[0], int(1.05 * g.size[1])))
    g2 = g1.crop((0, 0, g.size[0], g.size[1]))

    final_foreground = Image.merge('RGBA', (r2, g2, b, a))

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

def resize_with_padding(img, expected_size):
  img.thumbnail((expected_size[0], expected_size[1]))
  # print(img.size)
  delta_width = expected_size[0] - img.size[0]
  delta_height = expected_size[1] - img.size[1]
  pad_width = delta_width // 2
  pad_height = delta_height // 2
  padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
  return ImageOps.expand(img, padding)

def padding(img, expected_size):
  desired_size = expected_size
  delta_width = desired_size - img.size[0]
  delta_height = desired_size - img.size[1]
  pad_width = delta_width // 2
  pad_height = delta_height // 2
  padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
  return ImageOps.expand(img, padding)