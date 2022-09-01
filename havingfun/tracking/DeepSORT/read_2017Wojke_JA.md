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

## Introduction  
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
:cat: Overcome this issue by replacing the association metric with a more informed metric that combines motion and
appearance information 
