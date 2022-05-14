from PyQt5.QtWidgets import QApplication
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl
import numpy as np
import cv2
from BackgroundRemover import Remover
from BackgroundReplacer import  Replacer
from GlitchArt import GlitchEffect

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
    count = 0
    model = Remover()
    while success:
        #Getting Foreground Person
        result1 = model.predict(image)
        #Adding glitch effect on person
        result2 = GlitchEffect.Glitch(result1)

        result3 = Replacer.custom_background('tufta.jpg', result2)
        result4 = Replacer.custom_background('tufta.jpg', result1)

        #result.save('Output/image'+str(count)+'.png')


        #cv2.imwrite('Output\\image'+str(count)+'.png', result)

        frameArrFull.append(result3)
        frameArrBackgrnd.append(result4)
        count = count + 1
        success, image = vidcap.read()
        print('Read a new frame: ', success)
    return frameArrBackgrnd,frameArrFull

def convertFramesToVideo(frames,name):
    height, width, layers = frames[0].shape
    size = (width,height)
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(*'MPEG'), 30.0, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

if __name__ == '__main__':
    #we are reading the video firstl
    framesFull,framesBckgrnd = convertVideoToFrames('testingVideo.mp4')
    #then the frames that we read(and eventually passed to our model) reconverting to video
    convertFramesToVideo(framesFull,"tuftaVideo")
    convertFramesToVideo(framesBckgrnd,"TuftaVideo2")


