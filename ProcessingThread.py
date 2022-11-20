import os

from PyQt5.QtCore import QThread, QMutex, QTime, qDebug, QMutexLocker, pyqtSignal
from PyQt5.QtGui import QImage
from queue import Queue
import cv2
import numpy as np
import argparse
import imutils
import time
import cv2
import real_time_object_detection

from MatToQImage import matToQImage
from Structures import *
from Config import *


class ProcessingThread(QThread):
    newFrame = pyqtSignal(QImage)
    updateStatisticsInGUI = pyqtSignal(ThreadStatisticsData)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def __init__(self, sharedImageBuffer, deviceUrl, cameraId, parent=None):
        super(QThread, self).__init__(parent)
        self.sharedImageBuffer = sharedImageBuffer
        self.cameraId = cameraId
        self.detectCheker = False
        # Save Device Url
        self.deviceUrl = deviceUrl
        # Initialize members
        self.doStopMutex = QMutex()
        self.processingMutex = QMutex()
        self.t = QTime()
        self.processingTime = 0
        self.doStop = False
        self.enableFrameProcessing = False
        self.sampleNumber = 0
        self.fpsSum = 0.0
        self.fps = Queue()
        self.currentROI = QRect()
        self.imgProcFlags = ImageProcessingFlags()
        self.imgProcSettings = ImageProcessingSettings()
        self.statsData = ThreadStatisticsData()
        self.frame = None
        self.currentFrame = None


    def run(self):
        while True:
            ##############################
            # Stop thread if doStop=True #
            ##############################
            self.doStopMutex.lock()
            if self.doStop:
                self.doStop = False
                self.doStopMutex.unlock()
                break
            self.doStopMutex.unlock()
            ################################
            ################################

            # Save processing time
            self.processingTime = self.t.elapsed()
            # Start timer (used to calculate processing rate)
            self.t.start()

            with QMutexLocker(self.processingMutex):
                # Get frame from queue, store in currentFrame, set ROI
                # self.currentFrame = Mat(self.sharedImageBuffer.getByDeviceUrl(self.deviceUrl).get().clone(),
                #                         self.currentROI)
                self.currentFrame = self.sharedImageBuffer.getByDeviceUrl(self.deviceUrl).get()[
                                    self.currentROI.y():(self.currentROI.y() + self.currentROI.height()),
                                    self.currentROI.x():(self.currentROI.x() + self.currentROI.width())].copy()

                # Example of how to grab a frame from another stream (where Device Url=1)
                # Note: This requires stream synchronization to be ENABLED (in the Options menu of MainWindow)
                #       and frame processing for the stream you are grabbing FROM to be DISABLED.
                # if sharedImageBuffer.containsImageBufferForDeviceUrl(1):
                #     # Grab frame from another stream (connected to camera with Device Url=1)
                #     Mat frameFromAnotherStream = Mat(sharedImageBuffer.getByDeviceUrl(1).getFrame(), currentROI)
                #     # Linear blend images together using OpenCV and save the result to currentFrame. Note: beta=1-alpha
                #     addWeighted(frameFromAnotherStream, 0.5, currentFrame, 0.5, 0.0, currentFrame)

                ##################################
                # PERFORM IMAGE PROCESSING BELOW #
                ##################################

                # Grayscale conversion (in-place operation)
                if self.imgProcFlags.grayscaleOn and (
                        self.currentFrame.shape[2] == 3 or self.currentFrame.shape[2] == 4):
                    self.currentFrame = cv2.cvtColor(self.currentFrame, cv2.COLOR_BGR2GRAY)

                # Smooth (in-place operations)
                if self.imgProcFlags.smoothOn:
                    if self.imgProcSettings.smoothType == 0:
                        # BLUR
                        self.currentFrame = cv2.blur(self.currentFrame,
                                                     (self.imgProcSettings.smoothParam1,
                                                      self.imgProcSettings.smoothParam2))
                    elif self.imgProcSettings.smoothType == 1:
                        # GAUSSIAN
                        self.currentFrame = cv2.GaussianBlur(self.currentFrame,
                                                             (self.imgProcSettings.smoothParam1,
                                                              self.imgProcSettings.smoothParam2),
                                                             sigmaX=self.imgProcSettings.smoothParam3,
                                                             sigmaY=self.imgProcSettings.smoothParam4)
                    elif self.imgProcSettings.smoothType == 2:
                        # MEDIAN
                        self.currentFrame = cv2.medianBlur(self.currentFrame, self.imgProcSettings.smoothParam1)

                # Dilate
                if self.imgProcFlags.dilateOn:
                    self.currentFrame = cv2.dilate(self.currentFrame, self.kernel,
                                                   iterations=self.imgProcSettings.dilateNumberOfIterations)
                # Erode
                if self.imgProcFlags.erodeOn:
                    self.currentFrame = cv2.erode(self.currentFrame, self.kernel,
                                                  iterations=self.imgProcSettings.erodeUrlOfIterations)
                # Flip
                if self.imgProcFlags.flipOn:
                    self.currentFrame = cv2.flip(self.currentFrame, self.imgProcSettings.flipCode)
                # Canny edge detection
                if self.imgProcFlags.cannyOn:
                    self.currentFrame = cv2.Canny(self.currentFrame,
                                                  threshold1=self.imgProcSettings.cannyThreshold1,
                                                  threshold2=self.imgProcSettings.cannyThreshold2,
                                                  apertureSize=self.imgProcSettings.cannyApertureSize,
                                                  L2gradient=self.imgProcSettings.cannyL2gradient)
                if self.imgProcFlags.carsOn:
                    if (self.detectCheker == False):
                        ap = argparse.ArgumentParser()
                        ap.add_argument('prototxt', nargs='?', default="MobileNetSSD_deploy.prototxt.txt")
                        ap.add_argument('model', nargs='?', default="MobileNetSSD_deploy.caffemodel")
                        ap.add_argument("-c", "--confidence", type=float, default=0.2,
                                        help="minimum probability to filter weak detections")
                        args = vars(ap.parse_args())

                        # initialize the list of class labels MobileNet SSD was trained to
                        # detect, then generate a set of bounding box colors for each class
                        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                   "sofa", "train", "tvmonitor"]
                        COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

                        # load our serialized model from disk
                        net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt',
                                                       'MobileNetSSD_deploy.caffemodel')
                        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
                        self.detectCheker = True

                    # initialize the video stream, allow the cammera sensor to warmup,
                    # and initialize the FPS counter

                    # loop over the frames from the video stream
                    # grab the frame from the threaded video stream and resize it
                    # to have a maximum width of 400 pixels
                    image = self.currentFrame
                    image = imutils.resize(image, width=400)

                    # grab the frame dimensions and convert it to a blob
                    (h, w) = image.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                                 0.007843, (300, 300), 127.5)

                    # pass the blob through the network and obtain the detections and
                    # predictions
                    net.setInput(blob)
                    detections = net.forward()

                    # loop over the detections
                    for i in np.arange(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with
                        # the prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by ensuring the `confidence` is
                        # greater than the minimum confidence
                        if confidence > args["confidence"]:
                            # extract the index of the class label from the
                            # `detections`, then compute the (x, y)-coordinates of
                            # the bounding box for the object
                            idx = int(detections[0, 0, i, 1])
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # draw the prediction on the frame
                            label = "{}: {:.2f}%".format(CLASSES[idx],
                                                         confidence * 100)
                            cv2.rectangle(image, (startX, startY), (endX, endY),
                                          COLORS[idx], 2)
                    self.currentFrame = image






                ##################################
                # PERFORM IMAGE PROCESSING ABOVE #
                ##################################

                # Convert Mat to QImage
                self.frame = matToQImage(self.currentFrame)

                # Inform GUI thread of new frame (QImage)
                self.newFrame.emit(self.frame)

            # Update statistics
            self.updateFPS(self.processingTime)
            self.statsData.nFramesProcessed += 1
            # Inform GUI of updated statistics
            self.updateStatisticsInGUI.emit(self.statsData)

        qDebug("Stopping processing thread...")

    def doShowImage(self, val):
        with QMutexLocker(self.processingMutex):
            self.doShow = val

    def updateFPS(self, timeElapsed):
        # Add instantaneous FPS value to queue
        if timeElapsed > 0:
            self.fps.put(1000 / timeElapsed)
            # Increment sample number
            self.sampleNumber += 1

        # Maximum size of queue is DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH
        if self.fps.qsize() > PROCESSING_FPS_STAT_QUEUE_LENGTH:
            self.fps.get()

        # Update FPS value every DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH samples
        if self.fps.qsize() == PROCESSING_FPS_STAT_QUEUE_LENGTH and self.sampleNumber == PROCESSING_FPS_STAT_QUEUE_LENGTH:
            # Empty queue and store sum
            while not self.fps.empty():
                self.fpsSum += self.fps.get()
            # Calculate average FPS
            self.statsData.averageFPS = self.fpsSum / PROCESSING_FPS_STAT_QUEUE_LENGTH
            # Reset sum
            self.fpsSum = 0.0
            # Reset sample number
            self.sampleNumber = 0

    def stop(self):
        with QMutexLocker(self.doStopMutex):
            self.doStop = True

    def updateBoxesBufferMax(self, boxesBufferMax):
        with QMutexLocker(self.processingMutex):
            self.boxesBufferMax = boxesBufferMax

    def updateImageProcessingFlags(self, imgProcFlags):
        with QMutexLocker(self.processingMutex):
            self.imgProcFlags.grayscaleOn = imgProcFlags.grayscaleOn
            self.imgProcFlags.smoothOn = imgProcFlags.smoothOn
            self.imgProcFlags.dilateOn = imgProcFlags.dilateOn
            self.imgProcFlags.erodeOn = imgProcFlags.erodeOn
            self.imgProcFlags.flipOn = imgProcFlags.flipOn
            self.imgProcFlags.cannyOn = imgProcFlags.cannyOn
            self.imgProcFlags.carsOn = imgProcFlags.carsOn

    def updateImageProcessingSettings(self, imgProcSettings):
        with QMutexLocker(self.processingMutex):
            self.imgProcSettings.smoothType = imgProcSettings.smoothType
            self.imgProcSettings.smoothParam1 = imgProcSettings.smoothParam1
            self.imgProcSettings.smoothParam2 = imgProcSettings.smoothParam2
            self.imgProcSettings.smoothParam3 = imgProcSettings.smoothParam3
            self.imgProcSettings.smoothParam4 = imgProcSettings.smoothParam4
            self.imgProcSettings.dilateNumberOfIterations = imgProcSettings.dilateNumberOfIterations
            self.imgProcSettings.erodeUrlOfIterations = imgProcSettings.erodeUrlOfIterations
            self.imgProcSettings.flipCode = imgProcSettings.flipCode
            self.imgProcSettings.cannyThreshold1 = imgProcSettings.cannyThreshold1
            self.imgProcSettings.cannyThreshold2 = imgProcSettings.cannyThreshold2
            self.imgProcSettings.cannyApertureSize = imgProcSettings.cannyApertureSize
            self.imgProcSettings.cannyL2gradient = imgProcSettings.cannyL2gradient

    def setROI(self, roi):
        with QMutexLocker(self.processingMutex):
            self.currentROI.setX(roi.x())
            self.currentROI.setY(roi.y())
            self.currentROI.setWidth(roi.width())
            self.currentROI.setHeight(roi.height())

    def getCurrentROI(self):
        return QRect(self.currentROI.x(), self.currentROI.y(), self.currentROI.width(), self.currentROI.height())


