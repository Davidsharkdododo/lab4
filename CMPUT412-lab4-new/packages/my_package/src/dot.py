#!/usr/bin/env python3
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage
from duckietown_msgs.msg import WheelsCmdStamped, WheelEncoderStamped
from duckietown.dtros import DTROS, NodeType

class LaneControllerNode(DTROS):
    def __init__(self, node_name):
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        
        # Get vehicle name and topics.
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self.wheels_topic = f"/{self._vehicle_name}/wheels_driver_node/wheels_cmd"
        
        # Control parameters.
        self.controller_type = "P"
        self.Kp = -0.25
        self.Ki = 0.025
        self.Kd = 0.05
        self.base_speed = 0.4

        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = rospy.Time.now()
        
        # Camera calibration parameters.
        self.camera_matrix = np.array([[324.2902860459547, 0.0, 308.7011853118279],
                                       [0.0, 322.6864063251382, 215.88480909087127],
                                       [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.array([-0.3121956791769329, 0.07145309916644121,
                                      -0.0018668141393665327, 0.0022895877440351907, 0.0],
                                     dtype=np.float32)
        
        # Homography matrix (maps image coordinates to ground coordinates).
        self.homography = np.array([
            -0.00013679516037023445,  0.0002710547390276784,  0.32374273628358996,
            -0.0013732279193212306,  -3.481942844615056e-05,   0.43480445263628115,
            -0.0007393075649167115,   0.009592518288014648,    -1.1012483201073726
        ]).reshape(3, 3)
        
        # HSV ranges for lane detection.
        self.hsv_ranges = {
            "yellow": (np.array([20, 70, 100]), np.array([30, 255, 255])),
            "white": (np.array([0, 0, int(216*0.85)]), np.array([int(179*1.1), int(55*1.2), 255]))
        }
        # Flag to determine which lane color to use.
        self.use_yellow_lane = False
        
        # Variables to manage lane switching/dot detection cooldown.
        self.lane_switch_start_time = None   # Set when lane switch is triggered.
        self.lane_switch_cooldown = rospy.Duration(10)  # Total cooldown period during which dot detection is skipped.
        
        self.bridge = CvBridge()
        # Create publishers.
        self._publisher = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.undistorted_topic = f"/{self._vehicle_name}/camera_node/image/compressed/distorted"
        self.pub_undistorted = rospy.Publisher(self.undistorted_topic, Image, queue_size=1)
        self.lane_topic = f"/{self._vehicle_name}/camera_node/image/compressed/lane"
        self.pub_lane = rospy.Publisher(self.lane_topic, Image, queue_size=15)
        
        # Create a blob detector for duckiebot (circle grid) detection.
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.minArea = 10
        blob_params.minDistBetweenBlobs = 2
        self.simple_blob_detector = cv2.SimpleBlobDetector_create(blob_params)
        # Expected circle grid dimensions for the duckiebot pattern.
        self.circlepattern_dims = [7, 3]
        
        # Subscribe to the camera feed.
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        
        self.rate = rospy.Rate(2)
        self.log("Lane controller initialized.")

    def undistort_image(self, image):
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

    def preprocess_image(self, image):
        resized = cv2.resize(image, (640, 480))
        return cv2.GaussianBlur(resized, (5, 5), 0)

    def detect_lane_fast(self, image):
        """
        Optimized fast lane detection using fixed-height scanning.
        Scans for lane pixels (either white or yellow) at fixed heights in the bottom half.
        """
        full_height, width = image.shape[:2]
        half_height_start = full_height // 2
        bottom_half = image[half_height_start:full_height, :]
        
        # Choose lane color based on flag.
        lane_color = "yellow" if self.use_yellow_lane else "white"
        hsv = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2HSV)
        lane_mask = cv2.inRange(hsv, self.hsv_ranges[lane_color][0], self.hsv_ranges[lane_color][1])
        
        height = full_height - half_height_start
        center_x = width // 2
        
        # Define scan heights.
        scan_heights = [int(height * pct) for pct in [0, 0.10, 0.20, 0.40, 0.70]]
        output = image.copy()
        bottom_half_out = output[half_height_start:full_height, :]
        base_weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        
        offsets = []
        valid_detections = []
        
        for i, y in enumerate(scan_heights):
            for x in range(center_x, width, 3):
                if lane_mask[y, x] > 0:
                    offsets.append(x - center_x)
                    valid_detections.append(i)
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 255, 0), -1)
                    break
        
        error = None
        if offsets:
            bottom_idx = valid_detections.index(max(valid_detections))
            bottom_offset = offsets[bottom_idx]
            max_deviation = 10
            filtered_offsets = []
            filtered_detections = []
            for i, offset in enumerate(offsets):
                if offset - bottom_offset <= max_deviation:
                    filtered_offsets.append(offset)
                    filtered_detections.append(valid_detections[i])
                else:
                    detection_idx = valid_detections[i]
                    y = scan_heights[detection_idx]
                    x = center_x + offset
                    cv2.circle(bottom_half_out, (x, y), 3, (0, 0, 255), -1)
            if filtered_offsets:
                total_weight = sum([base_weights[i] for i in filtered_detections])
                normalized_weights = [base_weights[i] / total_weight for i in filtered_detections]
                weighted_offsets = [filtered_offsets[i] * normalized_weights[i] for i in range(len(filtered_offsets))]
                error_raw = sum(weighted_offsets)
                error = error_raw * 0.01 - 1.2
            color = (0, 0, 255) if len(valid_detections) < 3 else (0, 255, 0)
            cv2.putText(bottom_half_out, f"{len(valid_detections)}/5 pts, err:{error:.3f}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output, error

    def calculate_p_control(self, error, dt):
        return self.Kp * error

    def calculate_pd_control(self, error, dt):
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        rospy.loginfo("Derivative: %.3f", derivative)
        output = self.Kp * error + self.Kd * derivative
        self.prev_error = error
        return output

    def calculate_pid_control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def get_control_output(self, error, dt):
        ctrl_type = self.controller_type.upper()
        if ctrl_type == "P":
            return self.calculate_p_control(error, dt)
        elif ctrl_type == "PD":
            return self.calculate_pd_control(error, dt)
        elif ctrl_type == "PID":
            return self.calculate_pid_control(error, dt)
        else:
            rospy.logwarn("Unknown controller type '%s'. Using P controller.", self.controller_type)
            return self.calculate_p_control(error, dt)
        
    def publish_cmd(self, control_output):
        control_output = max(min(control_output, 0.3), -0.3)
        left_speed = self.base_speed - control_output
        right_speed = self.base_speed + control_output
        
        cmd_msg = WheelsCmdStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.vel_left = left_speed
        cmd_msg.vel_right = right_speed
        self._publisher.publish(cmd_msg)
    
    def turn_left_45(self):
        """
        Commands the robot to turn approximately 45 degrees to the left.
        Adjust the velocities and duration according to your robot's kinematics.
        """
        rospy.loginfo("Turning 45 degrees to the left.")
        turn_duration = 0.6  # Duration in seconds (adjust as needed)
        start_time = rospy.Time.now()
        rate = rospy.Rate(50)
        while rospy.Time.now() - start_time < rospy.Duration(turn_duration):
            cmd = WheelsCmdStamped()
            cmd.header.stamp = rospy.Time.now()
            # For a left turn: slower left wheel and faster right wheel.
            cmd.vel_left = -0.6
            cmd.vel_right = 0
            self._publisher.publish(cmd)
            rate.sleep()
        # Stop after turn.
        stop_cmd = WheelsCmdStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.vel_left = 0
        stop_cmd.vel_right = 0
        self._publisher.publish(stop_cmd)
    
    def callback(self, msg):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec() if self.last_time else 0.1
        self.last_time = current_time
        
        # Before processing, check cooldown.
        if self.lane_switch_start_time is not None:
            elapsed_since_switch = current_time - self.lane_switch_start_time
            # After 4 seconds, revert lane color back to white.
            if elapsed_since_switch >= rospy.Duration(4.0) and self.use_yellow_lane:
                rospy.loginfo("4 seconds elapsed. Switching back to white lane.")
                self.use_yellow_lane = False
            # Skip dot detection until cooldown expires.
            if elapsed_since_switch < self.lane_switch_cooldown:
                dot_detection_active = False
            else:
                self.lane_switch_start_time = None
                dot_detection_active = True
        else:
            dot_detection_active = True
        
        # Convert the image.
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        undistorted = self.undistort_image(cv_image)
        preprocessed = self.preprocess_image(undistorted)
        
        # --- Duckiebot detection logic ---
        if dot_detection_active:
            (detection, centers) = cv2.findCirclesGrid(
                undistorted,
                patternSize=tuple(self.circlepattern_dims),
                flags=cv2.CALIB_CB_SYMMETRIC_GRID,
                blobDetector=self.simple_blob_detector,
            )
            if detection > 0 and centers is not None and centers.shape[0] >= 2:
                dot0 = centers[0, 0]
                dot1 = centers[1, 0]
                distance_dots = np.linalg.norm(dot0 - dot1)
                rospy.loginfo("Distance between first two dots: %.2f pixels", distance_dots)
                # Trigger lane switch if distance exceeds threshold.
                if distance_dots > 10:
                    rospy.loginfo("Distance > 13 detected. Stopping for 3 seconds, turning left 45°, and switching lane detection.")
                    stop_cmd = WheelsCmdStamped()
                    stop_cmd.header.stamp = rospy.Time.now()
                    stop_cmd.vel_left = 0
                    stop_cmd.vel_right = 0
                    self._publisher.publish(stop_cmd)
                    rospy.sleep(3)
                    # Turn left 45 degrees.
                    self.turn_left_45()
                    # Immediately switch lane detection to yellow.
                    self.use_yellow_lane = True
                    self.lane_switch_start_time = rospy.Time.now()
        
        # --- Lane following logic ---
        output, error = self.detect_lane_fast(preprocessed)
        lane_msg = self.bridge.cv2_to_imgmsg(output, encoding="bgr8")
        self.pub_lane.publish(lane_msg)
        
        if error is not None:
            control_output = self.get_control_output(error, dt)
            self.publish_cmd(control_output)

if __name__ == '__main__':
    node = LaneControllerNode(node_name='lane_controller_node')
    rospy.spin()
