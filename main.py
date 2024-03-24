import numpy as np
import cv2
import vision_definitions
import math
from naoqi import ALProxy
import motion
import utils

ON_NAO = False

if ON_NAO:
    # Initialize connection to the robot
    port = 9559
    robot_ip = "192.168.0.103"
    tts = ALProxy("ALTextToSpeech", robot_ip, port)
    tts.say("Hi")

    # ----------------------------------------------------------------------------------------------------
    # Vision
    # ----------------------------------------------------------------------------------------------------

    # Camera setup
    cameraProxy = ALProxy("ALVideoDevice", robot_ip, port)
    camera_index = 0
    resolution = vision_definitions.kVGA
    colorSpace = vision_definitions.kBGRColorSpace
    fps = 30

    # Subscribe to camera
    cameraName = cameraProxy.getCameraName(camera_index)
    videoClient = cameraProxy.subscribeCamera(cameraName, camera_index, resolution, colorSpace, fps)

    def get_image():
        frame =  cameraProxy.getImageRemote(videoClient)
        frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frame[1], frame[0], frame[2]])
        return frameArray
    
    # ----------------------------------------------------------------------------------------------------
    # Camera Position
    # ----------------------------------------------------------------------------------------------------
    
    try:
        motionProxy = ALProxy("ALMotion", robot_ip, port)
    except:
        print("Could not create proxy to ALMotion")

    # Example showing how to get the position of the top camera
    name            = "CameraTop"
    space           = motion.FRAME_ROBOT
    useSensorValues = True

    def extrinsic_matrix():
        result = motionProxy.getTransform(name, space, useSensorValues)
        result = np.array(result)
        result = result.reshape(4, 4)
        return result

else:
    def extrinsic_matrix():
        return np.eye(4)

    VIDEO = True
    if VIDEO:
        import time

        i = 0
        def get_image():
            global i

            time.sleep(0.5)
            image = cv2.imread('./data/video2/image_' + str(i % 49) + '.png')
            i += 1
            return image

    else:
        def get_image():
            return cv2.imread('./data/test_2.png')

def detect_blue_ball(image):
    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_range = np.array([95, 100, 75])
    high_range = np.array([115, 255, 255])
    mask = cv2.inRange(hue_image, low_range, high_range)
    eroded = cv2.erode(mask, None, iterations=1)
    dilated = cv2.dilate(eroded, None, iterations=1)
    gaus = cv2.GaussianBlur(dilated, (7, 7), 3.5)
    circles = cv2.HoughCircles(gaus, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=15, minRadius=15, maxRadius=100)
    cv2.imshow('mask', gaus)
    return circles if circles is not None else []

def draw_circles(image, circle):
    x, y, radius = circle
    center = (x, y)
    cv2.circle(image, center, radius, (0, 255, 0), 2)
    
def calculate_ball_position(circle, real_diameter, K, RT):
    x, y, radius = circle
    diameter_in_pixels = 2 * radius
    distance = (real_diameter * K[0, 0]) / diameter_in_pixels

    # Convert image coordinates to camera coordinates
    _x_camera = (x - K[0, 2]) * distance / K[0, 0]
    _y_camera = (y - K[1, 2]) * distance / K[1, 1]
    _z_camera = distance

    x_camera = _z_camera
    y_camera = -_x_camera
    z_camera = -_y_camera

    # Camera coordinates
    point_camera = np.array([x_camera, y_camera, z_camera, 1])  # Use homogeneous coordinates

    # Apply the extrinsic matrix to convert to world coordinates
    # Assuming RT is the [R|T] matrix extended to 4x4 with a row [0, 0, 0, 1]
    point_world = np.dot(RT, point_camera)

    # Since we used homogeneous coordinates, divide by the last element if not 1
    if point_world[3] != 0 and point_world[3] != 1:
        point_world = point_world / point_world[3]

    # Now point_world contains the position in the world coordinates
    return point_world[:3]  # Return only x, y, z components, omitting the homogeneous coordinate

# From http://doc.aldebaran.com/1-14/family/robots/video_robot.html
def intrinsic_matrix():
    image_w = 640
    image_h = 480
    fov_w = np.deg2rad(60.9)
    fov_h = np.deg2rad(47.6)

    f_x = (image_w / 2) / np.tan(fov_w / 2)
    f_y = (image_h / 2) / np.tan(fov_h / 2)
    c_x = image_w / 2
    c_y = image_h / 2

    K = np.array([
        [f_x,     0,      c_x],
        [0,       f_y,    c_y],
        [0,       0,      1  ]
    ])

    # Calibrated parameters
    # K = np.array([
    #     [670.83636187,   0.,         336.29166902],
    #     [0.,         668.11148516, 253.29365806],
    #     [ 0.,           0.,           1.  ]
    # ])
    
    return K

def undistort(img, intrinsic_matrix,
              dist_coeffs=np.array([[0.22678288, -0.54274653, 0.01301456, 0.01108197, 0.37626486]])):
    return cv2.undistort(img, intrinsic_matrix, dist_coeffs, None, intrinsic_matrix)


ball_diameter = 0.055
K = intrinsic_matrix()
RT = extrinsic_matrix()

try:
    while True:
        frame = get_image()
        frame = undistort(frame, K)
        # utils.save_image_to_disk('./data/video/', frame)
        circles = detect_blue_ball(frame)
        if len(circles) > 0:
            circle = circles[0][0]
            draw_circles(frame, circle)
            ball_position = calculate_ball_position(circle, ball_diameter, K, RT)
            # print(K)

            print("Ball position:", ball_position)
        else:
            print("No ball detected")

        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
            break
finally:
    if ON_NAO:
        cameraProxy.unsubscribe(videoClient)  # Unsubscribe after loop exit
    cv2.destroyAllWindows()
