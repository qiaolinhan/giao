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
Use the (squared) Mahalanobis distance between predicted Kalman states and newly arrived measurements:
$$d^{(1)}(i , j) = (d_j - y_i)^TS_i^{-1}(d_j-y_i)$$  
Denote the projection of the $i$-th track distribution into 
