# coding: utf-8

#import matplotlib.pyplot as plt
# An example using startStreams

import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame

try:
    from pylibfreenect2 import OpenCLPacketPipeline
    pipeline = OpenCLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

enable_rgb = True
enable_depth = True

fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1)

serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

types = 0
if enable_rgb:
    types |= FrameType.Color
if enable_depth:
    types |= (FrameType.Ir | FrameType.Depth)
listener = SyncMultiFrameListener(types)

# Register listeners
device.setColorFrameListener(listener)
device.setIrAndDepthFrameListener(listener)

if enable_rgb and enable_depth:
    device.start()
else:
    device.startStreams(rgb=enable_rgb, depth=enable_depth)

# NOTE: must be called after device.start()
if enable_depth:
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())

undistorted = Frame(512, 424, 4)
registered = Frame(512, 424, 4)

kernelweights=[14,12,10,8,6,4,2,1,1,1]
kernel=[]
for i in range(len(kernelweights)):
    kernel.append(np.ones((kernelweights[i],kernelweights[i]),np.float32)/(kernelweights[i]*kernelweights[i]))



while True:
    frames = listener.waitForNewFrame()

    if enable_rgb:
        color = frames["color"]
    if enable_depth:
        ir = frames["ir"]
        depth = frames["depth"]

    if enable_rgb and enable_depth:
        registration.apply(color, depth, undistorted, registered)
    elif enable_depth:
        registration.undistortDepth(depth, undistorted)

    #if enable_depth:
        #cv2.imshow("ir", ir.asarray() / 65535.)
        #cv2.imshow("depth", depth.asarray() / 4500.)
        
        #print("maxdepth", depth.asarray().max()/4500.)
        #print("mindepth", depth.asarray().min()/4500.)
        #cv2.imshow("undistorted", undistorted.asarray(np.float32) / 4500.)
    if enable_rgb:
        #INIT DPB STUFF
        deparray=np.floor(depth.asarray()/450.)
        #deparray=depth.asarray()/4500.
        
        dzero = (deparray == 0.).astype(int) 
        done = (deparray == 1.).astype(int) 
        dtwo = (deparray == 2.).astype(int) 
        dthree = (deparray == 3.).astype(int) 
        dfour = (deparray == 4.).astype(int) 
        dfive = (deparray == 5.).astype(int) 
        dsix = (deparray == 6.).astype(int) 
        dseven = (deparray == 7.).astype(int) 
        deight = (deparray == 8.).astype(int) 
        dnine = (deparray == 9.).astype(int) 
        
        
        imgarraycolor=cv2.resize(color.asarray(),(int(1920/3.75),int(1080/2.545)))
       
        imgarray = cv2.cvtColor(imgarraycolor,cv2.COLOR_BGR2GRAY)

        izero = cv2.filter2D(imgarray,-1,kernel[0]) 
        ione = cv2.filter2D(imgarray,-1,kernel[1]) 
        itwo = cv2.filter2D(imgarray,-1,kernel[2]) 
        ithree = cv2.filter2D(imgarray,-1,kernel[3]) 
        ifour = cv2.filter2D(imgarray,-1,kernel[4]) 
        ifive = cv2.filter2D(imgarray,-1,kernel[5]) 
        isix = cv2.filter2D(imgarray,-1,kernel[6]) 
        iseven = cv2.filter2D(imgarray,-1,kernel[7]) 
        ieight = cv2.filter2D(imgarray,-1,kernel[8]) 
        inine = cv2.filter2D(imgarray,-1,kernel[9]) 

        #print(dfive.shape)
        #print(ifive.shape)
        
        
        #mfive=cv2.bitwise_and(ifive,dfive)

        composite= izero*dzero + ione*done + itwo*dtwo + ithree*dthree + ifour*dfour + ifive*dfive + isix*dsix + iseven*dseven + ieight*deight + inine*dnine
        composite2=composite.astype(np.dtype('uint8'))        
        #print("imgarray")        
        #print(imgarray.dtype)
        #print(imgarray.min())
        #print(imgarray.mean())
        #print("composite2")
        #print(composite2.dtype)
        #print(composite2.min())
        #print(composite2.mean())

        cv2.imshow("color",cv2.resize(composite2,(1024,848)))
        #cv2.imshow("color",composite2)
        #kernel = np.ones((40,40),np.float32)/1600
        #gauss1 = cv2.filter2D(color.asarray(),-1,kernel)
        #cv2.imshow("color", cv2.resize(gauss1,
                                       #(int(1920 / 3.75), int(1080 / 2.545))))
    #if enable_rgb and enable_depth:
        #cv2.imshow("registered", registered.asarray(np.uint8))

    listener.release(frames)

    key = cv2.waitKey(delay=1)
    if key == ord('q'):
        break

device.stop()
device.close()

sys.exit(0)
