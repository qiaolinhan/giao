# READ: Simple Online and Realtime Tracking with a Deep Association Metric  
2017Wojke_JA  
Simple online and realtime tracking (SORT)  

* multiple object tracking with a focus on simple , effective algorithms  
* Integrete appearance information to improve the performance of SORT $\Rightarrow$ Be able to track objects through
  longer periods of occlusions  
* effectively reducing the number of identity switches  
1) Offline: learn a deep association metric on a large-scale person re-identification dataset  
2) Online: Establish measurement-to-track associations using nearest neighbor queries in visual apperance space.  
3) Experimental evaluation shows: the extensions reduce the number of identity switches by 45%, high frame rate.  

## 1. Introduction  
Tracking-by-detection $\Rightarrow$ leading paradigm in multiple object tracking.  
Object trajectories $\Rightarrow$ global optimization problem $\Rightarrow$ processes entire video batches at once.  

### Popular frameworks
* flow network formulations
* Probabilistic graphical models  

:x: Due to batch processing $\Rightarrow$ not applicable in online scenarios where a target identity must be available
at each time step.  

### Traditional methods  
* Multiple Hypothesis Tracking (MHT)
* Joint Probabilistic Data Association Filter (JPDAF)  

These methods perform data association on a frame-by-frame basis.  
:x: The performnace of these methods comes at increased computational and impementation complexity.  

### SORT  
* A much simpler framework
* Performs Kalman filtering in image space
* Frame-by-frame data association using **Hungarian method** with an association metric the measures bounding box
  overlap.  

:o: Simple approach achieves favorable performance at high frame rates.  

while achieving overall good performance in terms of tracking precision and accuracy, SORT returns a relatively high
  number of identity switches.  
**Reason:** The employed association metric is only accurate when state estimation uncertainty is low.  
:x: A deficiency in traking through occlusions as they typically appear in frontal-view  camera scenes.  

#### SORT $\Rightarrow$ DeepSORT
:cat: Overcome this issue by replacing the association metric with a more informed metric that combines motion and
appearance information.  
* Apply a convolutional neural network (CNN) that has been trained to discriminate pedestrains on a large-scale person
   re-identification dataset.  
* Through integration of this network, increase robustness against misses and occlusions
   while keeping the system easy to implement, efficient, and applicable to online scenarios.

## 2. DeepSORT
Adopt a conventional single hypothesis tracking methodology with recursive Kalman Filtering and frame-to-frame data
association.
### 2.1 Track handling and state estimation  
track handling and Kalman filtering framework is mostly identical to the original formulation in [12].  
`Assume: a camera is uncalibrated and have no ego-motion information available`  
These circumstances pose a challenge to the filtering framework, it is the most common setup considered in recent
multiple object tracking benchmark.  
Therefore, our tracking scenario is defined on the eight dimensional state space $(u, v, \gamma, h, \dot{x}, \dot{y},
\dot{\gamma}, \dot{h})$.  
* bounding box center position $(u, v)$
* aspect ratio $\gamma$
* height $h$  

Use a standard Kalman filter with constant velocity motion and linear observation model. Take the bounding coordinates
$(u, v, \gamma, h)$ as direct observations of the object state.  
For each track $k$, count the number of frames since the last successful measurement association $a_k$.  
Tracks that exceed a predefined maximum age $A_{max}$ are considered to have left the scene and are deleted from the trackset.
#### 2.2 Assignment Problem
*******************
* indicator 指标  
* association 关联
* un-accounted camera 未标定的相机
* appearance descriptor 外观描述符
* admissible 可接受
* cascade 级联
* consequently 一致地
* Intuitively 直观地
* counterintuitively 反直觉地
* *****************

Use the (squared) Mahalanobis distance between predicted Kalman states and newly arrived measurements:
$$d^{(1)}(i , j) = (d_j - y_i)^TS_i^{-1}(d_j-y_i)$$  
* An indicator $b_{i, j}^{(1)}$
* If association between the $i$-th track and $j$-th detection is admissible  
* The corresponding Mahalanobis threshold is $t^{(1)} = 9.4877$  

:o: Mahalanobis distance is a suitable association metric when motion uncertainty is low.  

**The predicted state distribution obtained from the Kalman Filtering framework provides only a rough estimate of the
object location.** $\Rightarrow$ A second metric into the assignment problem.  

* Each bounding box detection $d_j$, compute an appearance descriptor $r_i$ with $||r_j|| = 1$
* Keep a gallery $\mathcal{R}_k$  

:o: The second metric measures the smallest cosine distance between the $i$-th track and $j$-th detection in appearance
space.  

* Again, introduce a binary variable to indicate if an association is admissible according to $b_{i, j}^{(2)}$  

Apply a pre-trained CNN to compute bounding box appearance descriptors. (in Section 2.4)  

### 2.3 Matching Cascade  
Instead of solving for measurement-to-track associations in a global assignment problem, a cascade that solves a series
of subproblems.  
Consider the following situation: An object is occluded for a longer period of time, subsequent Kalman filter
predictions increase the uncertainty associated with the object location.

* Consequently, probability mass spreads out in state space and the observation likelihood becomes less peaked.  
* Intitively, association metric should account for this spread of probability mass by increasing the
  measurement-to-track distance.  
* Counterintuitively, when two track comptete for the same detection, the Mahalanobis distance favors larger uncertainty,
  because it effectively reduces the distance in standard deviations of any detection towards the projected track mean.

#### Matching Algorithm
* Input: the setof track $\mathcal{T}$ and detection $\mathcal{D}$ indices as well as the maximum age $A_{max}$
* Compute association cost matrix and the matrix of admissible associations.  
* Iterate over track age $n$ to solve a linear assignment problem for tracks of increasing age $n$ to solve a linear
  assignment problem for tracks of increasing age.  
### 2.4 Deep Appearance Descriptor
*******************
* to this end 为此
* While the details of our training procedure are out of scope of this paper
* *****************
:o: Simple nearest neighbor queries without additional metric learning.  
Reqires: a well-discriminating feature embedding to be trained offline, before the actual online tracking application.  
:o: Employ a CNN (**A wide residual network: 2 Conv layers + 6 residual blocks + dense**) pre-trained, making it well suited for deep metric learning in a people tracking context.  
:o: 2,800,864 parameters and one forward pass of 32 bounding boxes takes about 30ms on GTX1050 GPU.  

## 3. Experiment
**********************
* We assess the performance of our tracker on the MOT16 benchmark
* Evaluation is acarried out according to the following metrics
* assess 评估
* life span 寿命
**********************

* Mostly tracked (MT): Percentage of ground-trurh tracks that have the same label for at leat 80% of their life span.
* Identity switches (ID): Number of times the reported identity of a grounf-truth track changes.  
* Fragmentation (FM): Number of times a track is interrupted by a missing detection.

:o: The adaptions successfully reduce the number of identity switches. In comparision to SORT, ID switches reduce from
1423 to 781. This is a decrease of approximately 45%.  
:o: The DeepSORT method is also a strong competitor to other online tracking frameworks. In particular, returns fewest
number of identity switches of all online methods while maintaining competitive MOTA scores, track fragmentations, and
false negatives.

### 4. Conclusion
Presented an extension to SORT that **incorporates appearance information through a pre-trained association metric.**  
:o: Able to track though longer periods of occlusion, making SORT a strong competitor to state-of-art online tracking
algorithms.  
:o: Remains simple to implement and runs in real time.

