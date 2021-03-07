import os
import numpy as np
import cv2 as cv
import time
import argparse
from src.tracker import *
from src.model_specific import *

if not os.path.exists('output'):
    os.makedirs('output')


winName = 'IVT Tracker demo'
drawnBox = np.zeros(4)
initialBox = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False
def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, initialBox, initialize
    if event == cv.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    initialBox = drawnBox.copy()
    initialBox[[0,2]] = np.sort(initialBox[[0,2]])
    initialBox[[1,3]] = np.sort(initialBox[[1,3]])


if __name__ == "__main__":
    print("[INFO] Program started")

    # Parse command line params
    parser = argparse.ArgumentParser(description='Incremental visual tracker.')
    parser.add_argument('-i', '--input', metavar='Input', type=str, help='input file')
    parser.add_argument('-d', '--debug', metavar='Debug', type=int, help='do show debug', default = 0)
    parser.add_argument('-r', '--record', metavar='Record', type=int, help='do record', default = 0)
    parser.add_argument('-t', '--test', metavar='Test', type=int, help='do test', default = 0)
    args = parser.parse_args()

    File = None
    if args.test == 1:
        try:
            np.random.seed(0) # <- for testing only
            File = open("tests/matlab-data.txt", "r") # <- for testing
        except:
            print("[INFO] Skip tests")

    # Create tracker
    tracker = IncrementalTracker(
        dof = DOF,
        affsig = AFFSIG, 
        nsamples = NSAMPLES,
        condenssig = CONDENSSIG, 
        forgetting = FORGETTING, 
        batchsize = BATCHSIZE, 
        tmplShape = (TMPLSIZE, TMPLSIZE), 
        maxbasis = MAXBASIS, 
        errfunc = 'L2'
    )

    # Init cv-window
    cv.namedWindow(winName, cv.WINDOW_NORMAL)
    cv.setMouseCallback(winName, on_mouse, 0)

    # get first frame
    capture = cv.VideoCapture(args.input)
    if args.test == 1: # <- for testing
        for _ in range(299):
            capture.read()

    writer = None
    ret, frame0 = capture.read()
    frame0 = cv.resize(frame0, (0, 0), None, RESIZE_RATE, RESIZE_RATE)
    cv.resizeWindow(winName, frame0.shape[1], frame0.shape[0])

    frameNum = 0
    Error = 0.0
    while ret and capture.isOpened():
        if frameNum > 0:
            ret, frame = capture.read()
            frame = cv.resize(frame, (0, 0), None, RESIZE_RATE, RESIZE_RATE)
        else:
            frame = frame0
        if not ret:
            print("[INFO] Video ended")
            break
        
        if INITIAL_BOX is None:
            while frameNum == 0 and not initialize:
                drawImg = frame.copy()
                cv.putText(drawImg, "Draw box around target object", (10, 20), cv.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255))
                cv.rectangle(drawImg,
                        (int(initialBox[0]), int(initialBox[1])),
                        (int(initialBox[2]), int(initialBox[3])),
                        [0,0,255], 2)
                cv.imshow(winName, drawImg)
                cv.waitKey(1)

        # -------------------- CORE -------------------- #
        startTime = time.time()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray) / 255

        # do tracking
        if frameNum == 0:
            if INITIAL_BOX is None: # <- debug
                w = initialBox[2] - initialBox[0]
                h = initialBox[3] - initialBox[1]
                cx = initialBox[0] + int(w/2)
                cy = initialBox[1] + int(h/2)
                box = np.array([cx, cy, w, h], dtype = np.float32)
            else:
                box = INITIAL_BOX
            est = tracker.track(gray, box)
        else:
            est = tracker.track(gray)

        tmpl = tracker.getTemplate()
        param = tracker.getParam()

        endTime = (time.time() - startTime) * 1000
        # -------------------- //// -------------------- #

        # --------------------- VIZ -------------------- #
        if args.debug:
            debugFrame = makeDetailedFrame(frameNum, frame, tmpl, param, (TMPLSIZE, TMPLSIZE), endTime)
            cv.resizeWindow(winName, debugFrame.shape[1], debugFrame.shape[0])
            cv.imshow(winName, debugFrame)
            if writer is None and args.record:
                writer = cv.VideoWriter('output/output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20, (debugFrame.shape[1], debugFrame.shape[0]))
            if args.record:
                writer.write(debugFrame)
        else:
            drawEstimatedRect(frame, param, (TMPLSIZE, TMPLSIZE))
            cv.resizeWindow(winName, frame.shape[1], frame.shape[0])
            cv.imshow(winName, frame)
            if writer is None and args.record:
                writer = cv.VideoWriter('output/output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20, (frame.shape[1], frame.shape[0]))
            if args.record:
                writer.write(frame)

        if File is not None:
            true = [ float(val) for val in File.readline().split() ]
            Error += np.linalg.norm(param['est'] - np.array(true))
            print(Error)
        # -------------------- //// -------------------- #

        frameNum += 1
        key = cv.waitKey(30)
        if key & 0xFF == ord('q') or key == 27:
            break
    
    if File is not None:
        File.close()
    if writer is not None:
        writer.release()
    capture.release()
    cv.destroyAllWindows()
    print("[INFO] Program successfully finished")