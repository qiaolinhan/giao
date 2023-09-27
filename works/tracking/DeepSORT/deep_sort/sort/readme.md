# Code representations  

`detection.py` save a box of object detection, and the confidence of this box, and the acquired fetures. And it gives
transfer schemes of different shapes of boxes.  
`iou_matching.py` Compute the IOU between two boxes.  
`kalman_filter.py` Using kalman filter to predict the trajectory of detectio boxes.  
`linear_assignment.py` Match hungarion algorithm to match predicted trajectory and the detection boxes.  
`nn_matching.py` Compute the Euclidean distance, cosin distance, _etc_ to compute the most near distance.  
`preprocessing.py` 非极大抑制， output the best detection boxes.  
`track.py` Storing the trajectory information, which includes position and speed information.  
`tracker.py` Storing all trajectory information, initialize the first frame, the predictions and updates of kalman
filter, cascade matching and IOU matching.
