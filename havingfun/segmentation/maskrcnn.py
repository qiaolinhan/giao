# image: PIL Image of size (H, W)
# target: a dict (dictionary) containing: 
#   boxes (FloatTensor[N, 4]), N bounding bexes, every box is [x0, y0, x1, y1]
#   labels (Int64Tensor[N]) 
#   image_id (Int64Tensor[1])
#   area (Tensor[N]) used during evaluation metric, to separate the metric scores between small, medium and large boxes
#   iscrod (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
#   masks, segmentation masks for eachone of the objects
#   keypoints