#functions necessary for the execution of the process
import base64
import cv2
import datetime
import logging
import numpy as np
import subprocess
import time
from lib.controller import *
from lib.cam_config import *
from fischertechnik.camera.VideoStream import VideoStream
from fischertechnik.controller.Motor import Motor
from lib.model_utils import *
import logging
from lib.display import *

from lib.TFLiteModel import TFLiteModel
from lib.ObjectDetector import ObjectDetector

#define positions
MovementSpeed = min(max(300, 1), 512)
position_camera = 105
position_white  = 195 
position_red    = 280
position_blue   = 360
position_fail   = 443
led.set_brightness(int(0))


cascade_path = '/opt/ft/workspaces/custom/stanger-circle-cascade.xml'
tf_model_path = '/opt/ft/workspaces/custom/efficient_net_v2.tflite'

#create model
model = TFLiteModel(tf_model_path)


label_dict = {1: 'white', 2: 'red', 3: 'blue'}
class_names = ["blue", "white", "red", "fail"]

#create color detector
detector = ObjectDetector(
    cascade_path, 
    minSize=(80,80), 
    label_dict=label_dict,
    minNeighbors=5,
)

#eject functions
def eject_red():
    compressor.on()
    logging.debug("RED")
    piston_eject_red.on()
    time.sleep(1)
    piston_eject_red.off()
    compressor.off()

def eject_white():
    compressor.on()
    logging.debug("WHITE")
    piston_eject_white.on()
    time.sleep(1)
    piston_eject_white.off()
    compressor.off()

def eject_blue():
    compressor.on()
    logging.debug("BLUE")
    piston_eject_blue.on()
    time.sleep(1)
    piston_eject_blue.off()
    compressor.off()

def eject_fail():
    compressor.on()
    logging.debug("FAIL")
    piston_eject_fail.on()
    time.sleep(1)
    piston_eject_fail.off()
    compressor.off()

#drive to positon
def drive_to_position(position):
    print("Drive to position: ", position)
    print("Counter: ", TXT_SLD_M_C1_motor_step_counter.get_count())
    SetBeltSpeedSteps(MovementSpeed, (position - position_camera) - (TXT_SLD_M_C1_motor_step_counter.get_count()))
    AwaitBeltToReachPosition()

def AwaitBeltToReachPosition():
    while True:
        if (not motor.is_running()):
            break
        time.sleep(0.010)

def SetBeltSpeedSteps(BeltSpeed, BeltSteps):
    if BeltSteps < 0:
        motor.set_speed(int(BeltSpeed), Motor.CW)
        motor.set_distance(int(BeltSteps * -1))
    else:
        motor.set_speed(int(BeltSpeed), Motor.CCW)
        motor.set_distance(int(BeltSteps))


#define take picture

def take_picture():
    print('taking picture')

    for i in range(512):
        led.set_brightness(i)
        time.sleep(0.001)    

    time.sleep(0.5)

    image = camera.read_frame()
    saveFileandPublish(image)

    led.set_brightness(int(0))

    return image

def process():
    while True:
        if(PartInGoodsReceipt()):

            # Reset interface
            reset_interface()
            displayGif()

            display.set_attr("part_pass_fail.text", str(containInHTML('i', 'processing')))

            # Move part to camera
            motor.set_speed(int(MovementSpeed * 0.5), Motor.CCW)
            motor.start_sync()
            for i in range(401):
                if not PartInGoodsReceipt():
                    break
                if i >= 399:
                    motor.stop_sync()
                    raise Exception("Insertion fault, workpiece did not clear the lightbeam in expected Time. [Trubleshoot:  Is the workpiece stuck somewhere?]")
                time.sleep(0.01)
            motor.stop_sync()

            SetBeltSpeedSteps(MovementSpeed, position_camera)
            AwaitBeltToReachPosition()

            # Start processing
            image = take_picture()
            prediction_image = image.copy()

            TXT_SLD_M_C1_motor_step_counter.reset()
            motor.set_speed(int(160), Motor.CCW)
            motor.set_distance(int(300))
            

            x, y, w, h = 0, 0, 0, 0

            # Detect color
            # Get the largest object in the image and its color
            largest_object, image = detector.get_largest_object("/opt/ft/workspaces/last-image.png")
            if largest_object is not None:
                x, y, w, h = largest_object
                cropped_image = detector.get_cropped_image(image, largest_object)
                cropped_image = cv2.resize(cropped_image, (100, 100))

                # Draw rectangle around the object
                rectangle_image = image
                cv2.rectangle(rectangle_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(rectangle_image, "Color: " + detector.detect_color(rectangle_image), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cropped_image = cv2.resize(image, (100, 100))

            detected_color = detector.detect_color(cropped_image)


            saveFileandPublish(image)

            # Detected color is str
            if detected_color in label_dict.values():
                detected_color = list(label_dict.keys())[list(label_dict.values()).index(detected_color)]
            else:
                raise ValueError("Detected color is not in label_dict")

            print("Color", detected_color)

            detected_color = np.array([detected_color], dtype=np.float32)

            # image = cv2.resize(image, (240, 240))
            prediction_image = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2RGB)

            prediction_image = np.float32(prediction_image)
            prediction_image = np.expand_dims(prediction_image, axis=0)

            # Predict result with model
            start_time = time.time()
            result = model.predict(detected_color, prediction_image)
            end_time = time.time()
            print(result) 
            elapsed_time = end_time - start_time
            print("Time taken for prediction:", elapsed_time, "seconds")    

            # Result
            # [[2.0026142e-02 5.1270140e-06 2.0064439e-07 9.7996855e-01]]

            # Set certainty
            display.set_attr("cRed.text", str(containInHTML('b', str(round(result[0][2] * 100, 2)) + "%")))
            display.set_attr("cBlue.text", str(containInHTML('b', str(round(result[0][0] * 100, 2)) + "%")))
            display.set_attr("cWhite.text", str(containInHTML('b', str(round(result[0][1] * 100, 2)) + "%")))
            display.set_attr("cFail.text", str(containInHTML('b', str(round(result[0][3] * 100, 2)) + "%")))

            result = np.argmax(result)
            result = class_names[result]

            print(result)

            display.set_attr("prediction_status.text", str(containInHTML("b", "Time: " + str(round(elapsed_time, 2)) + "s")))
           
            if result == 'blue':
                display.set_attr("blue.active", str(True).lower())
                display.set_attr("part_pass_fail.text", str(containInHTML('b', "Workpiece <font color='#88ff88'> PASSED</font>")))

                # SetBeltSpeedSteps(MovementSpeed, (position_blue - position_camera) - (TXT_SLD_M_C1_motor_step_counter.get_count()))
                drive_to_position(position_blue)
                eject_blue()
            elif result == 'fail':
                display.set_attr("fail.active", str(True).lower())
                display.set_attr("part_pass_fail.text", str(containInHTML('b', "Workpiece <font color='#ff8888'>FAILED</font>")))

                # SetBeltSpeedSteps(MovementSpeed, (position_fail - position_camera) - (TXT_SLD_M_C1_motor_step_counter.get_count()))
                drive_to_position(position_fail)
                eject_fail()
            elif result == 'red':
                display.set_attr("red.active", str(True).lower())
                display.set_attr("part_pass_fail.text", str(containInHTML('b', "Workpiece <font color='#88ff88'> PASSED</font>")))

                # SetBeltSpeedSteps(MovementSpeed, (position_red - position_camera) - (TXT_SLD_M_C1_motor_step_counter.get_count()))
                drive_to_position(position_red)
                eject_red()
            elif result == 'white':
                display.set_attr("white.active", str(True).lower())
                display.set_attr("part_pass_fail.text", str(containInHTML('b', "Workpiece <font color='#88ff88'> PASSED</font>")))

                # SetBeltSpeedSteps(MovementSpeed, (position_white - position_camera) - (TXT_SLD_M_C1_motor_step_counter.get_count()))
                drive_to_position(position_white)
                eject_white()

def reset_interface():
    display.set_attr("part_pass_fail.text", str(containInHTML('i', 'Not analysed yet')))
    display.set_attr("prediction_status.text", str(containInHTML("b", "Time: --")))
    display.set_attr("red.active", str(False).lower())
    display.set_attr("white.active", str(False).lower())
    display.set_attr("blue.active", str(False).lower())
    display.set_attr("fail.active", str(False).lower())

    # Reset certainty
    display.set_attr("cWhite.text", str(containInHTML('b', '')))
    display.set_attr("cBlue.text", str(containInHTML('b', '')))
    display.set_attr("cRed.text", str(containInHTML('b', '')))
    display.set_attr("cFail.text", str(containInHTML('b', '')))

def containInHTML(tag, value):
    return ''.join([str(x) for x in ['<', tag, '>', value, '</', tag, '>']])

def saveFileandPublish(image):
    global filename
    filename = '/opt/ft/workspaces/last-image.png'

    imageFitToDisplay = image
    imageFitToDisplay = cv2.resize(imageFitToDisplay, (240, 140))

    logging.debug("write png file: ", filename)

    cv2.imwrite(filename, imageFitToDisplay)

    subprocess.Popen(['chmod', '777', filename])

    with open(filename, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    imgb64 = "data:image/jpeg;base64," + (my_string.decode('utf-8'))
    time.sleep(0.2)

    # publish(imgb64,keytext,color,num,prob,duration)

    displaystr= "<img width='240' height='160' src='" +  imgb64  + "'>"
    display.set_attr("img_label.text", str(displaystr))



def displayGif():
    filename = "/opt/ft/workspaces/images/processing.jpg"

    # Read the image using OpenCV
    img = cv2.imread(filename)

    # Resize the image to the desired width and height
    width = 240
    height = 160
    resized_img = cv2.resize(img, (width, height))

    # Convert the resized image to base64
    retval, buffer = cv2.imencode('.jpg', resized_img)
    my_string = base64.b64encode(buffer)
    imgb64 = "data:image/jpeg;base64," + (my_string.decode('utf-8'))

    # Display the resized image
    displaystr = "<img width='240' height='160' src='" +  imgb64  + "'>"
    display.set_attr("img_label.text", str(displaystr))

def PartInGoodsReceipt():
    return part_at_start.is_dark()