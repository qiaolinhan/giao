# How DeepSORT works

## 1. Some comparision
### 1.1 Track by detection
* YOLOv4  
* Detectron2  
Disadvantage: Disturbance swoops in front of camera, the tracking will loose  

### 1.2 Traditional methods  
* Mean Shift  
* Optical Flow (Lucas Kanade)  
  * computationally complex
  * Prone to noise  
  * track lost during occlusion

### 1.3 SORT
* Kalman filter is acrucial compoments in DeepSORT
* SORT: Simple Online Realtime Tracking
The SORT comprises of 4 core components which are: Detection, Estimation, Target Association, 
#### 1.3.1 Detection  
Better Detection = Better Tracking  
800*600 image --> VGG16 --> 50*37*256 feature maps
#### 1.3.2 Estimation 
Detection --> Kalman filter  
E.g. For tracking rocket:  
  $$x = [u, v, s, r, \dot{u}, \dot{v},\dot{s}]^T$$
  where:  
* $u$: Horizontal pixel location of target center  
* $v$: Vertical pixel location of target center  
* $s$: Scale area of target bounding box  
* $r$: Aspect ratio of the target's bounding box
#### 1.3.3 Target Association  
Detection --> Kalman filter --> Target Association  
Each target's bounding box geometry is estimated by predicting its new location in the latest frame.  
The assignment cost matrix is then computed as the intersection-over-union (IOU) distance between each detection.  
$$IoU = \frac{Area of Overlap}{Area of Union}$$
#### 1.3.4 Track Identity Life Cycle
Detection --> Kalmanfilter --> target Association --> Target ID life cycle  
When objects enter and leave the image, unique identities need to be created or destroyed accordingly.  
<span style = "color:red"> Tlost frame</span>  

## 2. DeepSORT
CNN Object Detection --> SORT Object Tracking  
**Question: What makes DeepSORT so different?**
* Object detector that provides us detections
* The almighty Kalman filter tracking it and giving us missing tracks
* The Hungarain algorithm associates detections to tracked objects 

** Advantages of SORT **
* Overall good performance in terms of tracking precision and accuracy  
* Depite the effectiveness of Kalman filter, it returnsa relatively high number of identity switches  
* It has a deficiency in tracking throug occlusionsand different viewpoints  

**What is the difference between SORT and DeepSORT?**  
For DeepSORT, the authors introduced another distance metric based on the "appearance" of the objects.  
**Appearance Feature Vector**  
### 2.2 DeepSORT framework  
* a classifier is build based on dataset, which is trained meticulously until it achieves a reasonably good accuracy.  
* take this network and strip the final classification layer leaving behind a dense layer that produce a single feature
  vector, waititng to be classified.
* Use nearest neighbor queries in the visual appearance, this is to establish the measurement-to-track association (MTA),
  which is the process of determining the relation between a measurement and an existing track. ---- We also use the
  Mahalanobis distance as oppose to the euclidean distance for MTA.  
#### 2.2.1 DeepSORT Advantages  
* ID Switches reduced by 45%  
* High FPS  
#### 2.2.2 DeepSORT Alternatives  
| Models| Advantages| Challenges|
| ---| ---| ---|  
| Tracktor++| Accurate| Very slow|
| Track RCNN| Segmentation as bonus| very slow|
| JDE| Displayed decent performance of 12 FPS on average| Low resolusion|
| DeepSORT| * Fastest (16 FPS); * good accuracy; * Asolid choice for multiple object detection and tracking.| ---|  

**DeepSORT** is used for multiple object tracking  
* Need to do bounding box detections first and then tracker need to be combined with a detector  
```mermaid
graph LR
A(Object Detection)
B(Multi Object Tracking)
C(Kalman filtering: To process the correlation of frame-by-frame data)
D(Hunarian Algorithm: for correlation measurement)
E(CNN network: for training and feature extracting)
A --> B  
B --> C  
B --> D  
B --> E
```

:o: The high performance of DeepSORT is because its **Feature extracturion net**.
DeepSORT avoid much ID-switch because:
* The Feature extraction net extracted and saved the features inside the object detection box  
* Compare the features of the new came out object and the covered object which in previous frames $\rightarrow$ to find
  the previous object. obviously decreased the missing rate.

#### 2.2.3 DeepSORT steps 

程度, 级联， 匈牙利算法， 分配
Degree, Cascade, Hungarian Algorithm, Assignment

1. Capture original video frame  
2. USing object detector to detect objects in the original video frame  
3. Extract the features in the box of the detected objects. Where the features include: appearance features and motion
   feature (The motion features are for the convenience of Kalman filter prediction)  
4. Compute the matching level or range, and assign the IDs for each tracked object.

#### 2.2.4 SORT process 
The kernel concept of SORT (which is before DeepSORT) are Kalman Filter and Hungarian Algorithm  
:o: The function of Kalman Filter: To predict the motion in next frame based on this frame and previous frames. The first
detection result to initialize the motion of kalman filter.  
:o: The function of Hungarian Algorithm: To be brief, it is to solve th eproblem of assignment. Which is to assign some
detection boxes and kalman filter prediction boxes.  

```mermaid
graph LR
A(Detections)
B(IoU match)
C(Unmatched tracks)
D(Delet)
E(Unmatched Detections)
F(Matched Tracks)
G(New Tracks)
H(Tracks)
I(KF Predict)
J(KF Update)
A --> B  
B --> C  
B --> E
B --> F
C --> D
E --> G 
G --> H 
H --> I 
F --> J
J --> H
I --> B
```

#### 2.2.5 DeepSORT process 
SORT is relatively a shallow tracking algorithm. If the object is covered, the ID may miss. DeepSORT is an algorithm
which added **Matching Cascade**  and **Confirming New Trajectories** based on SORT.The tracks could be separated into states:  
* Confirmed  
* Unconformed  
The new tracks are in the sate of **Unconfirmed**  
The tracks in the sate of unconfirmed must match the detections continuously for times (Commonly, it is defult 3 times)
to transfer into the state of confirmed.  
The tracks in the state of confirmed must miss the matching for times (Commonly, it is defult 30 times) to be deleted.  
The process of DeepSORT could be stated as:  
```mermaid
graph LR
A(Detections)
B(Matching Cascade)
C(Unmatched tracks)
D(Unmatched Detections)
E(Matched Tracks)
F(IoU Match)
G(Unmatched Tracks)  
H(Unmatched Detections) 
I(Matched Tracks)
J(Unconformed) 
K(Confirmed)
L(New Tracks)
M(KF Update)
N(Delete)
O(> Max age)
P(< Max age)
Q(Tracks)
R(KF Predict)
S(Confirmed)
T(Unconfirmed)
A --> B  
B --> C  
B --> D
B --> E
D --> F 
F --> G  
F --> H  
F --> I  
G --> J  
G --> K  
H --> L
I --> M
E --> M
J --> N  
K --> O  
K --> P  
O --> N  
P --> Q  
L --> Q  
M --> Q
Q --> R  
R --> S  
R --> T  
S --> B  
T --> F
```
