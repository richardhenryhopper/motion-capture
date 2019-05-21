# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import io
import numpy as np
import threading
import imutils

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()

class MotionDetect():
        motionDetected = False
        movingObjectNum = 0
        motionFrame = []
        CONTOUR_MIN_AREA = 10000
        DELTA_THRESH = 30
        TIME_INTERVAL = 5
        frame = None
        thresh = None
        frameDelta = None
        triggerTime = 0
        
        def init_camera(self):
                camera.resolution = (640, 480) # Set resolution
                camera.framerate = 10
                camera.vflip = True # Flip image
        
        def fix_exposure(self):
                time.sleep(2) # Wait for the automatic gain control to settle
                camera.shutter_speed = camera.exposure_speed # Fix exposure values
                camera.exposure_mode = 'off' # Turn off auto exposure
                g = camera.awb_gains # Get gain settings
                camera.awb_mode = 'off' # Turn off auto white balance mode
                camera.awb_gains = g # Set gains

        def auto_exposure(self):
                camera.exposure_mode = 'on' # Turn on auto exposure
                camera.awb_mode = 'on' # Turn on auto white balance mode
                
        def capture_image(self):
                rawCapture = PiRGBArray(camera, size=(640, 480)) # Create picamera buffer
                camera.capture(rawCapture, format="bgr", use_video_port=True) # Capture raw image
                rawImage = rawCapture.array # Get numpy array
                grayImage = cv2.cvtColor(rawImage, cv2.COLOR_BGR2GRAY) # Converting to gray scale
                grayImage = cv2.GaussianBlur(grayImage, (21, 21), 0) # Gaussian blur 
                return(grayImage)

        def detect_motion(self):
                rawCapture = PiRGBArray(camera, size=(640, 480)) # Create picamera buffer
                avg = None
                time.sleep(0.1) # Allow the camera to warmup
                for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                        self.frame = f.array # Get numpy array
                        
                        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
                        gray = cv2.GaussianBlur(gray, (21, 21), 0) # Blur

                        # If the average frame is None, initialize it
                        if avg is None:
                                print("[INFO] starting background model...")
                                avg = gray.copy().astype("float")
                                rawCapture.truncate(0)
                                continue
                         
                        self.frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg)) # Compute the difference
                
                        self.thresh = cv2.threshold(self.frameDelta, self.DELTA_THRESH, 255, cv2.THRESH_BINARY)[1] # Threshold the delta image
                        self.thresh = cv2.dilate(self.thresh, None, iterations=2) # Dilate the thresholded image to fill in holes
                        (_, cnts, _) = cv2.findContours(self.thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find contours on thresholded image
                        
                        # Loop over the contours
                        self.motionDetected = False # Set flag
                        for c in cnts:
                                if cv2.contourArea(c) > self.CONTOUR_MIN_AREA: # Test size
                                        (x, y, w, h) = cv2.boundingRect(c) # Compute the bounding box
                                        cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Draw rectangle
                                        self.motionDetected = True # Set flag

                        #if self.motionDetected == False: # Check if background is static
                        cv2.accumulateWeighted(gray, avg, 0.5) # Accumulate the weighted average

                        rawCapture.truncate(0) # Clear the stream

        def read(self):
                frame = self.frame.copy() # Copy frame
                frameDelta = self.frameDelta.copy()
                return(frame, frameDelta)
                                                                                                
        def run(self):              
                motionDetectThread = threading.Thread(target=self.detect_motion)
                motionDetectThread.start()
                time.sleep(1)
     
                while(True): 
                        frame, frameDelta = self.read() # Read frame
                        cv2.imshow("Frame Delta", frameDelta) # Display delta frame
                        cv2.imshow("Motion Detection Feed", frame) # Display frame
                        if self.motionDetected == True:
                                if (time.time() - self.triggerTime) > self.TIME_INTERVAL:
                                        self.triggerTime = time.time() # Get trigger time
                                        print('Motion detected!')
                                self.motionDetected = False
                        
                        key = cv2.waitKey(1) & 0xFF # Break if 'q' key pressed
                        if key == ord("q"):
                                exit
                                
                motionDetectThread.stop()
                                
if __name__ == "__main__":
        motion = MotionDetect()
        motion.init_camera()
        motion.fix_exposure()
        motion.run()
        


 
	


        
