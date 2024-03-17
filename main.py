import numpy as np
import cv2
import vision_definitions
import math
from naoqi import ALProxy

ON_NAO = False

if ON_NAO:
    # Initialize connection to the robot
    tts = ALProxy("ALTextToSpeech", "192.168.0.102", 9559)
    tts.say("Hi")

    # Camera setup
    port = 9559
    robot_ip = "192.168.0.102"
    cameraProxy = ALProxy("ALVideoDevice", robot_ip, port)
    camera_index = 0
    resolution = vision_definitions.kVGA
    colorSpace = vision_definitions.kBGRColorSpace
    fps = 20

    # Subscribe to camera
    Name = cameraProxy.getCameraName(camera_index)
    videoClient = cameraProxy.subscribeCamera(Name, camera_index, resolution, colorSpace, fps)

    def get_image():
        frame =  cameraProxy.getImageRemote(videoClient)
        frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frame[1], frame[0], frame[2]])
        return frameArray

else:
    VIDEO = True
    if VIDEO:
        import time

        i = 0
        def get_image():
            global i

            time.sleep(1)
            image = cv2.imread('./data/video/image_' + str(i % 10) + '.png')
            i += 1
            return image

    else:
        def get_image():
            return cv2.imread('./data/test-img.png')

def detect_blue_ball(image):
    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_range = np.array([95, 150, 100])
    high_range = np.array([115, 255, 255])
    mask = cv2.inRange(hue_image, low_range, high_range)
    gaus = cv2.GaussianBlur(mask, (7, 7), 1.5)
    eroded = cv2.erode(gaus, None, iterations=2)
    dilated = cv2.dilate(eroded, None, iterations=2)
    circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=15, maxRadius=100)
    return circles if circles is not None else []

def draw_circles(image, circle):
    x, y, radius = circle
    center = (x, y)
    cv2.circle(image, center, radius, (0, 255, 0), 2)
    
def calculate_ball_position(circle, real_diameter, K):
    x, y, radius = circle
    diameter_in_pixels = 2 * radius
    distance = (real_diameter * K[0, 0]) / diameter_in_pixels

    # Convert image coordinates to camera coordinates
    x_camera = (x - K[0, 2]) * distance / K[0, 0]
    y_camera = (y - K[1, 2]) * distance / K[1, 1]
    z_camera = distance

    print(x, y)

    return np.array([x_camera, y_camera, z_camera])
    
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
    
    return K

ball_diameter = 0.05
K = intrinsic_matrix()

print(K)
# quit()

try:
    while True:
        frame = get_image()
        print(frame.shape)
        circles = detect_blue_ball(frame)
        if len(circles) > 0:
            circle = circles[0][0]
            draw_circles(frame, circle)
            ball_position = calculate_ball_position(circle, ball_diameter, K)
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
