# The states need to be considered  
Mean of $[c_x, c_y, r, h, v_x, v_y, v_r, v_h]$  
$(c_x, c_y)$: the centre  
$r$: width, height rate  
$h$: height  
Coveriance matrix: uncertainty of the target location information, which is a $8\times 8$ matrix

Every track need to predict the state of next time, and to adjust based on the detect result.

## Different Filters
To compare briefly, there could be a Table:

| 不同滤波与算子 | OpenCV Commands                                                                          | 高斯噪声               | 椒盐噪声            | 边界                            |
| ---            | ---                                                                                      | ---                    | ---                 | ---                             |
| 方盒滤波       | cv2.boxFilter(img, ddepth = -1, ksize = (5, 5), normalize = True)                        | 模糊                   | -                   | -                               |
| 均值滤波       | cv2.blur(img, (5, 5))                                                                    | 模糊                   | -                   | -                               |
| 高斯滤波       | cv2.GaussianBlur(img, (5, 5), sigmaX = 100)                                              | 去除高斯噪声，但会模糊 | 无用                | -                               |
| 中值滤波       | cv2.medianBlur(img, 5)                                                                   | -                      | 表现好， 几乎不模糊 | -                               |
| 双边滤波       | cv2.bilateralFilter(img, 7, sigmaColor = 20, sigmaSpace = 50)                            | 美颜效果               | 失效                | -                               |
| Sobel算子      | cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5); cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5) | -                      | -                   | 10 y方向，01 x方向              |
| Scharr算子     | cv2.Scharr(img, cv2.CV_64F, dx = 1, dy = 0); cv2.Scharr(img, cv2.CV_64F, dx = 0, dy = 1) | -                      | -                   | 只支持3*3 kernel， 擅长细小边缘 |




