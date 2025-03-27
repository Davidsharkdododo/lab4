# lab4

### Introduction

part1lane.py processes the camera images to detect AprilTags and programmed responses based on the tag type. The system recognizes Stop Signs, T-Intersections, and UofA Tags, responding with appropriate LED colors and stopping durations at intersections.

duck.py detects blue crosswalk lines and monitors for the presence of peDuckstrians. The Duckiebot stops briefly at empty crosswalks and waits longer when peDuckstrians are crossing.

dot.py detects obstacles in the path, stops at a safe distance to assess the situation, and then executes a sequence to navigate around the obstacle.

### Code structure overview

## part1lane.py:

Initialization:

Configures ROS topics (camera, wheels, LEDs, debug images), control parameters (PID gains, base speed), camera calibration, homography, and HSV ranges.

Sets up an AprilTag detector with tag-to-action mappings (e.g., red tag → stop 3 sec).

Initializes state variables for tag visibility, red line detection, and lane following.

Image Processing:

Undistort & Preprocess: Corrects lens distortion, resizes, and blurs images.

Lane Detection: Uses fixed-height scanning to identify lane pixels and compute a steering error.

Red Line Detection: Applies HSV thresholding to detect red lines and compute distance via homography.

Control:

Implements P/PD/PID control functions to compute steering corrections.

Publishes wheel commands and stops the robot when needed.

Updates LED patterns based on detected tag color.

Callback:

Processes incoming camera images, performing lane and red line detection.

Executes rate-limited AprilTag detection to overlay tag info, update LEDs, and retrieve stop instructions.

Uses detection states to decide whether to continue lane following or stop at an intersection.

Run Loop:

The node continuously runs using rospy.spin().

## duck.py:

Initialization:

Sets up ROS topics (camera, wheel encoders, commands), calibration, homography, and HSV ranges for blue and duck detection.

Initializes maneuver flags and a queue for detected lane colors.

Image Processing:

Methods for undistorting, resizing, blurring images, and detecting lane colors using HSV thresholds.

Computes distance using homography.

Encoder Callbacks:

Updates left and right wheel encoder ticks.

Motor Control:

Provides dynamic motor control to drive the robot a specified distance and stops the robot.

Camera Callback:

Processes each frame to detect lane colors and queues maneuvers for blue or duck detection.

Main Loop (run):

Executes maneuvers based on queued detections (stopping, waiting for duck clearance) or drives forward if no maneuvers are queued.

## dot.py:

Initialization:

Sets up ROS topics, control parameters, camera calibration, homography, and HSV ranges.

Initializes lane switching cooldown and a blob detector for circle grid (dot) detection.

Configures publishers (undistorted & annotated images) and subscribes to the camera feed.

Image Processing:

Undistorts, resizes, and blurs images.

Detects lane lines via fixed-height scanning to compute a lane error.

Control:

Implements P/PD/PID control functions to compute steering adjustments.

Publishes wheel commands and includes a routine to execute a 45° left turn.

Callback:

Processes incoming images to detect dots and trigger maneuvers (stop, turn, switch lane detection to yellow) if dot spacing exceeds a threshold.

Otherwise, follows the lane using computed error.


#### How to use it

### 1. Fork this repository

Use the fork button in the top-right corner of the github page to fork this template repository.


### 2. Build the program

After that open this folder in the terminal. Once you are in the folder, type in the terminal: dts devel build - H [Vehicle_name] -f to build the program in the duckiebot.


### 3. Run Part 1

To run the program, type in the terminal: dts devel run -H [Vehicle_name] -L part1lane. Replace [Vehicle_name] with your robot's name.


### 4. Run part 2

To run the program, type in the terminal: dts devel run -H [Vehicle_name] -L duck. Replace [Vehicle_name] with your robot's name.


### 5. Run part 3

To run the program, type in the terminal: dts devel run -H [Vehicle_name] -L dot. Replace [Vehicle_name] with your robot's name.
