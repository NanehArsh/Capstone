from PyQt5.QtWidgets import QApplication
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
import cv2
import time

from BackgroundRemover import Remover
from BackgroundReplacer import Replacer
from RGBGlitch import RGBGlitch
from Outliner import Outline

def runPlayer(filename):
    app = QApplication([])
    player = QMediaPlayer()
    wgt_video = QVideoWidget()  # Video display widget
    wgt_video.show()
    player.setVideoOutput(wgt_video)  # widget for video output
    player.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))  # Select video file
    player.play()
    app.exec_()

def convertVideoToFrames(filename):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    frameArrFull = []
    frameArrBackgrnd = []
    frameArrGlitch = []
    count = 0
    model = Remover()
    while success:

        # Getting Foreground Person
        masked_image,mask = model.predict(image)

        # Adding glitch effect on person
        masked_image_copy = masked_image.copy()
        outlined_image = Outline.MakeOutlline(masked_image_copy,mask)
        outlined_image = cv2.cvtColor(outlined_image, cv2.COLOR_RGBA2RGB)
        glitched_image = RGBGlitch.rgbglitch(masked_image)
        replaced_image = Replacer.custom_background('bg4.png', masked_image)
        # result.save('Output/image'+str(count)+'.png')
        #cv2.imwrite('PNG_Iframes_exported\\image'+str(count)+'.png', outlined_image)

        frameArrGlitch.append(glitched_image)
        frameArrBackgrnd.append(replaced_image)
        frameArrFull.append(outlined_image)
        #frameArrBackgrnd.append(result4)
        count = count + 1
        success, image = vidcap.read()
        print('Read a new frame: ', success)
    return frameArrBackgrnd, frameArrGlitch, frameArrFull

def convertFramesToVideo(frames, name):

    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25.0, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

if __name__ == '__main__':
    # we are reading the video firstly
    start = time.time()
    framesBckgrnd, framesRGB, frameArrOutline = convertVideoToFrames('test.mp4')
    finish = time.time()
    duration = round((finish - start))
    # then the frames that we read(and eventually passed to our model) reconverting to video
    convertFramesToVideo(framesBckgrnd, "test_background_replaced")
    convertFramesToVideo(framesRGB, "test_rgbglitch")
    convertFramesToVideo(frameArrOutline, "test_outline")
    #showing that video
    #runPlayer('output.mp4')

