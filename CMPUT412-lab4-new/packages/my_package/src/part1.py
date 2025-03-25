#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from dt_apriltags import Detector  # using dt_apriltags instead of apriltag
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped, LEDPattern
from std_msgs.msg import Header, ColorRGBA
from duckietown.dtros import DTROS, NodeType

# Wheel radius (m)
Wheel_rad = 0.0318

def compute_distance(ticks):
    rotations = ticks / 135
    return 2 * 3.1415 * Wheel_rad * rotations

def compute_ticks(distance):
    rotations = distance / (2 * 3.1415 * Wheel_rad)
    return rotations * 135

class ApriltagDetectionNode(DTROS):
    def __init__(self, node_name):
        super(ApriltagDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        self._left_encoder_topic = f"/{self._vehicle_name}/left_wheel_encoder_node/tick"
        self._right_encoder_topic = f"/{self._vehicle_name}/right_wheel_encoder_node/tick"
        self._ticks_left = 0
        self._ticks_right = 0
        self.drive = 1

        self.hsv_ranges = {
        "red": (np.array([0, 70, 150]), np.array([10, 255, 255]))
        }
        self.red_line_detected = False
        self.at_intersection = False
        
        # Variables to store tag detection state.
        self.last_tag_info = None  # Dict with keys "color" and "stop_duration"
        self.tag_visible = False   # True if a tag is detected in the current image
        self.tag_lost_time = None  # Time when tag was first lost

        # Camera calibration parameters.
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                      -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                     dtype=np.float32)
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        
        # Initialize the dt_apriltags detector for tag36h11 family.
        self.detector = Detector(families="tag36h11")
        self.last_detection_time = rospy.Time(0)
        self.detection_interval = 0.33  # seconds

        # Mapping: tag ID 21 is red (Stop Sign, 3 sec),
        #          tag ID 133 is blue (T-Intersection, 2 sec),
        #          tag ID 94 is green (UofA Tag, 1 sec).
        self.tag_mapping = {
            21: {"color": "red", "stop_duration": 3.0},
            22: {"color": "red", "stop_duration": 3.0},
            133: {"color": "blue", "stop_duration": 2.0},
            93: {"color": "green", "stop_duration": 1.0},
            94: {"color": "green", "stop_duration": 1.0}
        }
        # Default mapping if tag ID is not recognized.
        self.default_tag = {"color": "white", "stop_duration": 0.5}
        
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        # Publisher for the augmented image with apriltag annotations.
        self.pub_apriltag = rospy.Publisher(f"/{self._vehicle_name}/camera_node/image/compressed/apriltag", Image, queue_size=10)
        
        self._led_topic = f"/{self._vehicle_name}/led_emitter_node/led_pattern"
        self._led_pub = rospy.Publisher(self._led_topic, LEDPattern, queue_size=1)
        
        self.sub_left = rospy.Subscriber(self._left_encoder_topic, WheelEncoderStamped, self.callback_left)
        self.sub_right = rospy.Subscriber(self._right_encoder_topic, WheelEncoderStamped, self.callback_right)
        self.publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        
        self.rate = rospy.Rate(3)

    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        # Use the full image (crop here if desired).
        resized = cv2.resize(image, (640, 480))
        # Convert to grayscale to simplify detection.
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return gray, resized
    
    def detect_red_line(self, image):
        """
        Detects red lines in the image and returns whether a red line is detected
        and the computed distance from the line.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detected_distance = None
        red_line_detected = False
        
        lower, upper = self.hsv_ranges["red"]
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(c)
                u = x + w / 2
                v = y + h
                distance = self.compute_distance_homography(u, v)
                red_line_detected = True
                detected_distance = distance
                
                # Draw red line detection on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(image, f"Red line: {distance:.2f}m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return red_line_detected, detected_distance, image

    def compute_distance_homography(self, u, v):
        """
        Compute the distance to a point using homography.
        """
        point_img = np.array([u, v, 1.0])
        ground_point = self.homography @ point_img
        ground_point /= ground_point[2]
        X, Y = ground_point[0], ground_point[1]
        return np.sqrt(X**2 + Y**2)

    def update_leds(self, color_str):
        """
        Update LEDs based on the provided color string.
        Mapping:
            "red"   -> Stop Sign,
            "blue"  -> T-Intersection,
            "green" -> UofA Tag,
            "white" -> Default state.
        """
        num_leds = 5
        pattern_msg = LEDPattern()
        pattern_msg.header.stamp = rospy.Time.now()
        if color_str == "red":
            color = ColorRGBA(r=1, g=0, b=0, a=1)
        elif color_str == "blue":
            color = ColorRGBA(r=0, g=0, b=1, a=1)
        elif color_str == "green":
            color = ColorRGBA(r=0, g=1, b=0, a=1)
        else:  # "white" or unknown
            color = ColorRGBA(r=1, g=1, b=1, a=1)
        leds = [color for _ in range(num_leds)]
        pattern_msg.rgb_vals = leds
        self._led_pub.publish(pattern_msg)

    def callback_left(self, data):
        self._ticks_left = data.data

    def callback_right(self, data):
        self._ticks_right = data.data

    def dynamic_motor_control(self, left_power, right_power, distance_left, distance_right):
        """
        Drives the robot until the left and right wheels have traveled the specified distances.
        """
        rate = rospy.Rate(100)
        msg = WheelsCmdStamped()
        msg.vel_left = left_power
        msg.vel_right = right_power

        init_ticks_left = self._ticks_left
        init_ticks_right = self._ticks_right

        while (not rospy.is_shutdown() and
               abs(compute_distance(self._ticks_left - init_ticks_left)) < abs(distance_left) and
               abs(compute_distance(self._ticks_right - init_ticks_right)) < abs(distance_right)):

            # Optionally adjust speeds based on wheel distance errors.
            dist_ratio = distance_left / distance_right if distance_right != 0 else 1
            left_dist = self._ticks_left - init_ticks_left
            right_dist = self._ticks_right - init_ticks_right
            diff = abs(left_dist) - abs(right_dist) * dist_ratio
            modifier = 50 * diff / 1000  # Tune as needed.

            msg.vel_left = left_power * (1 - modifier)
            msg.vel_right = right_power * (1 + modifier)
            msg.header.stamp = rospy.Time.now()
            if(self.drive == 1): self.publisher.publish(msg)
            rate.sleep()

        # Stop the robot.
        stop_msg = WheelsCmdStamped()
        stop_msg.header = Header()
        stop_msg.header.stamp = rospy.Time.now()
        stop_msg.vel_left = 0
        stop_msg.vel_right = 0
        self.publisher.publish(stop_msg)

    def callback(self, msg):
        # Convert the incoming compressed image to an OpenCV image
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Apply camera undistortion
        undistorted = self.undistort_image(cv_image)
        
        # Convert to grayscale for preprocessing
        gray, color_image = self.preprocess_image(undistorted)
        
        # Check for red line detection
        red_line_detected, red_line_distance, annotated_image = self.detect_red_line(color_image)
        
        # Update red line detection state
        if red_line_detected and red_line_distance < 0.20:  # If within 20cm
            self.red_line_detected = True
        else:
            self.red_line_detected = False
        
        # Create a black and white version (binary threshold)
        _, bw_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Convert back to 3-channel (BGR) for annotation
        bw_annotated = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)
        
        # Enforce a detection rate limit
        if (rospy.Time.now() - self.last_detection_time) < rospy.Duration(self.detection_interval):
            return
        self.last_detection_time = rospy.Time.now()
        
        # Detect dt_apriltags in the grayscale image
        detections = self.detector.detect(gray, estimate_tag_pose=False)
        
        if detections:
            # Process the first detected tag
            detection = detections[0]
            corners = detection.corners.astype(int)
            pts = corners.reshape((-1, 1, 2))
            cv2.polylines(bw_annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            center = detection.center.astype(int)
            cv2.putText(bw_annotated, f"ID: {detection.tag_id}", tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            tag_info = self.tag_mapping.get(detection.tag_id, self.default_tag)
            self.update_leds(tag_info["color"])
            # Remember the tag info and reset loss timer
            self.last_tag_info = tag_info
            self.tag_visible = True
            self.tag_lost_time = None
            rospy.loginfo("Detected dt_apriltag ID: %d", detection.tag_id)
        else:
            self.tag_visible = False
            # If a previous tag exists, keep its color
            if self.last_tag_info is not None:
                self.update_leds(self.last_tag_info["color"])
            else:
                self.update_leds("white")
            if self.tag_lost_time is None:
                self.tag_lost_time = rospy.Time.now()
        
        # Add red line detection visualization to bw_annotated if needed
        # This is optional - you can keep the apriltag visualization separate
        
        # Publish the black and white annotated image
        bw_annotated_msg = self.bridge.cv2_to_imgmsg(bw_annotated, encoding="bgr8")
        self.pub_apriltag.publish(bw_annotated_msg)

    def run(self):
        # Start moving forward
        self.drive = 1  # Always start with driving enabled
        self.dynamic_motor_control(0.5, 0.5, 10, 10)
        
        while not rospy.is_shutdown():
            # Print debug info
            rospy.loginfo("Status: red_line=%s, tag_info=%s, at_intersection=%s, drive=%d", 
                        self.red_line_detected, 
                        "Present" if self.last_tag_info else "None",
                        self.at_intersection,
                        self.drive)
            
            # Priority 1: Check for red line detection with tag info
            if self.red_line_detected and not self.at_intersection and self.last_tag_info is not None:
                duration = self.last_tag_info["stop_duration"]
                color = self.last_tag_info["color"]
                rospy.loginfo("STOPPING AT RED LINE for %.1f seconds based on %s tag", duration, color)
                
                # Mark that we're at an intersection to prevent multiple stops
                self.at_intersection = True
                
                # Stop the robot
                self.drive = 0
                rospy.sleep(duration)
                
                # Resume driving
                self.drive = 1
                
                # Don't clear tag info yet - we might need it for LED colors
            
            # Reset intersection flag once we've passed the line
            elif not self.red_line_detected and self.at_intersection:
                rospy.loginfo("Passed intersection, resetting flags")
                self.at_intersection = False
                # Now clear tag info since we've completed the intersection
                self.last_tag_info = None
                self.tag_lost_time = None
            
            # Fallback only if we have no red line detection after ~2 seconds of tag loss
            elif (not self.red_line_detected and 
                self.last_tag_info is not None and 
                self.tag_lost_time is not None and 
                (rospy.Time.now() - self.tag_lost_time).to_sec() >= 2.0):
                
                duration = self.last_tag_info["stop_duration"]
                rospy.loginfo("NO RED LINE DETECTED after tag lost for 2+ seconds. Emergency stop for %.1f seconds", duration)
                
                # Stop the robot
                self.drive = 0
                rospy.sleep(duration)
                
                # Resume driving and clear tag info
                self.drive = 1
                self.last_tag_info = None
                self.tag_lost_time = None
            else:
                # Continue driving if none of the above conditions apply
                self.drive = 1
                
            self.rate.sleep()

if __name__ == '__main__':
    node = ApriltagDetectionNode(node_name='apriltag_detection_node')
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass