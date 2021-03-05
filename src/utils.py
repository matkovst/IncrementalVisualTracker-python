import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
colorMap = plt.get_cmap('jet')


def rad2deg(rad):
    return float((rad * 180) / np.pi)

def subimage(image, center, width, height, tmplsize, angle):
    
    shape = ( image.shape[1], image.shape[0] )
    deg = -rad2deg(angle) # since y-axis begins on top opencv requires negative angle for clockwise rotation
    M = cv.getRotationMatrix2D( center, deg, 1.0 )
    wimg = cv.warpAffine( image, M, shape )
    
    x1 = int( center[0] - width/2 )
    y1 = int( center[1] - height/2 )
    x1 = x1 if x1 >= 0 else 0
    y1 = y1 if y1 >= 0 else 0
    wimg = wimg[ y1:y1+height, x1:x1+width ]

    wimg = cv.resize(wimg, (tmplsize, tmplsize), None, cv.INTER_LINEAR, cv.BORDER_CONSTANT, 0)
    return wimg

def warpimg(img, p, sz):
    if not all(sz):
        sz = img.shape
    if len(p.shape) == 1:
        p = np.reshape(p, (1, p.size))
    tmplsize = sz[0]
    
    center = (p[:, 0], p[:, 1])
    angle = p[:, 3]
    width = int(p[:, 2]*tmplsize)
    height = int(width * p[:, 4])
    return subimage(img, center, int(width), int(height), tmplsize, angle)

def warpimgs(img, p, sz):
    if not all(sz):
        sz = img.shape
    if len(p.shape) == 1:
        p = np.reshape(p, (1, p.size))
    tmplsize = sz[0]
    
    nsamples = p.shape[0]
    center = (p[:, 0], p[:, 1])
    angle = p[:, 3]
    width = ( p[:, 2] * tmplsize )
    height = ( p[:, 4] * width )
    wimgs = np.zeros((tmplsize, tmplsize, nsamples))
    for i in range(nsamples):
        wimgs[:,:,i] = subimage(img, (int(center[0][i]), int(center[1][i])), int(width[i]), int(height[i]), tmplsize, angle[i])
    return wimgs

def convert(img, target_type_min, target_type_max, target_type):
    imin = np.min(img)
    imax = np.max(img)

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def makeDetailedFrame(fno, frame, tmpl, param, patchSize, timeElapsed = 0.0):

    originalFrame = frame.copy()
    detailedFrame = np.zeros(frame.shape, dtype = np.uint8)
    if 'param' in param:
        for i in range(param['param'].shape[0]):
            cx, cy = param['param'][i, 0], param['param'][i, 1]
            prob = param['conf'][i]
            if math.isnan(prob):
                prob = 0.0
            colorizedProb = np.array( colorMap( int(prob * 255) )[:-1][::-1] ) * 255
            cv.circle(detailedFrame, (cx, cy), 1, colorizedProb)
    hOffset = 20
    cv.putText(originalFrame, str(frame.shape[1]) + 'x' + str(frame.shape[0]), (10, 1*hOffset), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255))
    cv.putText(detailedFrame, "frame no: " + str(fno), (10, 1*hOffset), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 127, 0))
    cv.putText(detailedFrame, "time (ms): " + str(round(timeElapsed)), (10, 2*hOffset), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 127, 0))
    cv.putText(detailedFrame, "particles: " + str(param['param'].shape[0]), (10, 3*hOffset), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 127, 0))
    cv.putText(detailedFrame, "eigenbasis: " + str(tmpl['basis'].shape[1] if tmpl['basis'] is not None else 0), (10, 4*hOffset), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 127, 0))
    
    rrectParam = param['est']
    rrectW = rrectParam[2]*patchSize[0]
    rrectH = rrectW*rrectParam[4]
    rrect = ((rrectParam[0], rrectParam[1]), (rrectW, rrectH), -rad2deg(rrectParam[3]))
    rrectBox = np.int0( cv.boxPoints(rrect) )
    cv.drawContours(originalFrame, [rrectBox], 0 ,(0, 0, 255), 2)
    cv.drawContours(detailedFrame, [rrectBox], 0 ,(255, 127, 0), 2)

    mean = np.reshape( tmpl['mean'], patchSize )
    wimg = param['wimg']
    err = param['err'] if 'err' in param else np.zeros(patchSize)
    recon = param['recon'] if 'recon' in param else np.zeros(patchSize)
    mean = np.uint8(255 * mean)
    wimg = np.uint8(255 * wimg)
    err = convert(err, 0, 255, np.uint8)
    recon = np.uint8(255 * recon)
    stacked = np.hstack((mean, wimg, err, recon))
    stacked = cv.cvtColor(stacked, cv.COLOR_GRAY2BGR)
    stacked = cv.resize(stacked, (0, 0), None, 3, 3)
    #cv.putText(stacked, str('mean'), (5, 1*hOffset), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255))

    # if tmpl['basis'] is not None:
    #     mag = 3
    #     basisLen = tmpl['basis'].shape[1]
    #     basisPatches = np.zeros((patchSize[0], patchSize[1], basisLen), dtype = np.float32)
    #     eigenfacesFrame = np.reshape( tmpl['basis'][:, 0], patchSize )
    #     for i in range(1, basisLen):
    #         basisPatches[:, :, i] = np.reshape( tmpl['basis'][:, i], patchSize )
    #     for i in range(1, basisLen):
    #         eigenfacesFrame = np.hstack( (eigenfacesFrame, basisPatches[:, :, i]) )
    #     cv.imshow('Basis', cv.resize(eigenfacesFrame, (0, 0), None, 3, 3))

    finalFrameW = frame.shape[1]*2 if frame.shape[1]*2 > stacked.shape[1] else stacked.shape[1]
    finalFrameH = frame.shape[0]*2 if frame.shape[0]*2 > stacked.shape[0] else stacked.shape[0]
    finalFrame = np.zeros((finalFrameH, finalFrameW, 3), dtype = np.uint8)
    finalFrame[0:originalFrame.shape[0], 0:originalFrame.shape[1], :] = originalFrame
    finalFrame[0:detailedFrame.shape[0], originalFrame.shape[1]:originalFrame.shape[1]+detailedFrame.shape[1] :] = detailedFrame
    finalFrame[originalFrame.shape[0]:originalFrame.shape[0]+stacked.shape[0], 0:stacked.shape[1], :] = stacked

    return finalFrame

def drawEstimatedRect(img, param, patchSize):
    rrectParam = param['est']
    rrectW = rrectParam[2]*patchSize[0]
    rrectH = rrectW*rrectParam[4]
    rrect = ((rrectParam[0], rrectParam[1]), (rrectW, rrectH), rrectParam[3])
    rrectBox = np.int0( cv.boxPoints(rrect) )
    cv.drawContours(img, [rrectBox], 0 ,(0, 0, 255), 2)