# lab4

### Introduction

part1lane.py processes the camera images to detect AprilTags and programmed responses based on the tag type. The system recognizes Stop Signs, T-Intersections, and UofA Tags, responding with appropriate LED colors and stopping durations at intersections.

duck.py detects blue crosswalk lines and monitors for the presence of peDuckstrians. The Duckiebot stops briefly at empty crosswalks and waits longer when peDuckstrians are crossing.

dot.py detects obstacles in the path, stops at a safe distance to assess the situation, and then executes a sequence to navigate around the obstacle.

### Code structure overview

Class Definition:

AprilTagLaneController inherits from DTROS and encapsulates functionality for lane following and intersection management using AprilTags.

Initialization (__init__):

Sets up ROS topics, PID control parameters, camera calibration, homography, and HSV ranges for detecting lanes and red lines.

Initializes an AprilTag detector (rate-limited to one detection every 0.33 seconds) and maps tag IDs to specific LED colors and stop durations.

Creates publishers for wheel commands, LEDs, and debug images, and subscribes to the compressed camera feed.

Image Processing & Detection:

Undistortion & Preprocessing: Methods to correct lens distortion and prepare images (resizing, blurring, grayscale conversion).

Lane Detection: Uses fixed-height scanning with color thresholding to compute a lane offset error.

Red Line Detection: Applies HSV masking to identify red lines, compute distance using homography, and annotate the image.

AprilTag Detection: Runs at a lower rate, overlays tag information on images, and updates LED states based on detected tags.

Control & Actuation:

Contains PID (or P/PD) control methods to compute steering corrections from lane error.

Publishes wheel commands to adjust speed and direction.

Provides a method to stop the robot and update LED patterns based on tag instructions.

Callback & Main Loop:

Processes incoming camera images, runs detection algorithms, and handles decision logic to switch between lane following and stopping at intersections.

Uses ROS’s rospy.spin() in the run method to keep the node active.




## How to use it

### 1. Fork this repository

Use the fork button in the top-right corner of the github page to fork this template repository.


### 2. Adjust HSV range for your color lanes

Before running the code, please navigate to CMPUT412-lab3/packages/my_package/src to run color.py. Make changes to the HSV range for your color lanes if necessary.


### 3. Build the program

After that open this folder in the terminal. Once you are in the folder, type in the terminal: dts devel build - H [Vehicle_name] -f to build the program in the duckiebot.


### 4. Run Part 1

To run the program, type in the terminal: dts devel run -H [Vehicle_name] -L part1lane. Replace [Vehicle_name] with your robot's name.


### 5. Run part 2

To run the program, type in the terminal: dts devel run -H [Vehicle_name] -L duck. Replace [Vehicle_name] with your robot's name.


### 6. Run part 3

To run the program, type in the terminal: dts devel run -H [Vehicle_name] -L dot. Replace [Vehicle_name] with your robot's name.
