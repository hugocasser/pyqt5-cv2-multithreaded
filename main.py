import sys

from PyQt5.QtWidgets import QApplication
import qdarkstyle

from MainWindow import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()

    # Setup style
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    # Show main window
    win.show()

    # Start event loop
    sys.exit(app.exec_())
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
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


if __name__ == '__main__':
    main()
