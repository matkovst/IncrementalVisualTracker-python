import numpy as np
import cv2 as cv
import time
import argparse
from src.tracker import *
from src.model_specific import *
np.random.seed(0) # <- for testing only


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

# python demo.py -i C:/Users/Stanislav/Videos/david_indoor.avi -r 160 106 62 78 -0.02 -d 1
# python demo.py -i C:/Users/Stanislav/Videos/Greece.mp4 -r 170 110 52 78 -0.02 -d 1 -z 0.1
if __name__ == "__main__":
    print("[INFO] Program started")

    # Parse command line params
    parser = argparse.ArgumentParser(description='Incremental visual tracker.')
    parser.add_argument('-i', '--input', metavar='Input', type=str, help='input file')
    parser.add_argument('-r', '--rect', metavar='Rotated rect', type=float, nargs='+', help='Object location on first frame')
    parser.add_argument('-z', '--resize', metavar='Resize', type=float, help='Resize rate', default = 1.0)
    parser.add_argument('-d', '--debug', metavar='Debug', type=int, help='do show debug')
    args = parser.parse_args()
    resizeRate = args.resize

    File = open("tests/python-est-data.txt", "r") # <- for testing

    # Create tracker
    tracker = IncrementalTracker(
        affsig = affsig, 
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
    for _ in range(299):
        capture.read()

    writer = None
    ret, frame0 = capture.read()
    frame0 = cv.resize(frame0, (0, 0), None, resizeRate, resizeRate)

    frameNum = 0
    Error = 0.0
    while ret and capture.isOpened():
        if frameNum > 0:
            ret, frame = capture.read()
            frame = cv.resize(frame, (0, 0), None, resizeRate, resizeRate)
        else:
            frame = frame0
        if not ret:
            print("[INFO] Video ended")
            break
        
        while frameNum == 0 or mousedown:
            drawImg = frame.copy()
            cv.putText(drawImg, "Draw box around target object", (10, 20), cv.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255))
            if mouseupdown:
                frameNum += 1
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

        if frameNum == 1:
            w = initialBox[2] - initialBox[0]
            h = initialBox[3] - initialBox[1]
            cx = initialBox[0] + int(w/2)
            cy = initialBox[1] + int(h/2)
            box = np.array([cx, cy, w, h], dtype = np.float32)
            if INITIAL_BOX is not None: # <- debug
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
            if writer is None:
                writer = cv.VideoWriter('output/output.avi', cv.VideoWriter_fourcc('M','J','P','G'), 20, (debugFrame.shape[1], debugFrame.shape[0]))
            writer.write(debugFrame)
            cv.resizeWindow(winName, debugFrame.shape[1], debugFrame.shape[0])
            cv.imshow(winName, debugFrame)
        else:
            drawEstimatedRect(frame, param, (TMPLSIZE, TMPLSIZE))
            cv.resizeWindow(winName, frame.shape[1], frame.shape[0])
            cv.imshow(winName, frame)

        true = [ float(val) for val in File.readline().split() ]
        Error += np.linalg.norm(param['est'] - np.array(true))
        print(Error)
        # -------------------- //// -------------------- #

        frameNum += 1
        key = cv.waitKey(30)
        if key & 0xFF == ord('q') or key == 27:
            break
    
    File.close()
    if writer is not None:
        writer.release()
    capture.release()
    cv.destroyAllWindows()
    print("[INFO] Program successfully finished")