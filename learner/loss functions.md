学习一下这个loss function。 因为发现segmentation中，自己造的light U-net使用CEloss要么loss贼大，要么loss不下降。找不到原因，神烦。  
ok doge  
## 1. What is lossfunction
用来度量模型预测值，predection， 和真实值， label， 的差异程度的运算函数。  
* 是一个非负实函数。 通常表示为 $L(Y, f(x))$  
* 损失函数越小，模型鲁棒性越好
## 2. Some loss functions
### 2.1 基于距离的损失函数
输入数据映射到-基于距离度量的特征空间-（欧氏空间， 汉明空间， 等）  
映射后的样本看作空间上的点， 采用合适的损失函数度量特征空间上-样本真值-和-模型预测值-之间的距离。  
特征空间上两点距离越小-模型预测性能越好
#### 2.1.1 Mean squae error loss (MSE Loss)
eq: $$L(Y|f(x)) = \frac{1}{n} \sum_{i = 1}^{N}{(Y_i - f(x_i))^2}$$  
对于回归： MSE-度量-样本点到回归曲线的距离。通常被作为模型的-经验损失-或-算法的性能指标。    
优点： 无参数， 计算成本低， 具有明确物理意义。  
不足： 图像-语音处理-表现弱。  
但是： 仍然是评价信号质量的标准。  
coding：  
``` # python
import numpy as np
def MSELoss(x:list, y:list):
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    loss = np.sum(np.square(x - y)) / len(x)
    return loss

# torch
y_true = torch.tensor(y)
y_pred = torch.tensor(x)
mse_fc = torch.nn.MSELoss(y_true, y_pred)
mse_loss = mse_fc(x, y)
```  
#### 2.1.2 L2 loss
eq: $$L(Y|f(x)) = \sqrt{\frac{1}{n}\sum_{i = 1}^{N}{(Y_i - f(x_i))^2}}$$  
又称为-欧氏距离。 常用的距离度量方法，通常度量数据点之间的相似程度。  
优点：凸性， 可微性， 独立同分布且高斯噪声下-能提供最大似然估计。  
所以： 回归问题， 模式识别， 图像处理-最常用损失函数。  
coding：  
```
import numpy as np
def L2Loss(x:list, y: list):
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    loss = np.sqrt(np.sum(np.square(x - y)) / len(x))
    return loss
```  
#### 2.1.3 L1 loss
eq: $$L(Y|f(x)) = \sum_{i = 1}^{N}{|Y_i - f(x_i)|}$$  
又称-曼哈顿距离，表示残差的-绝对值-的和。   
advantage： 对离群点有很好的鲁棒性。    
disadvantage： 残差为零处不可导。更新的梯度始终相同- == 很小的损失值梯度也很大 - 不利于模型收敛    
针对收敛问题的解决办法： 使用变化的学习率， 损失接近最小值时降低学习率。  
coding：  
```
import numpy as np
def L1Loss(x: list, y: list):
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    loss = np.sum(np.abs(x - y)) / len(x)
    return loss
```  
#### 2.1.4 Smoooth L1 Loss
eq: <img src = 'https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29+%3D++%5Cbegin%7Bcases%7D+++++%5Cfrac%7B1%7D%7B2%7D%28Y-f%28x%29%29%5E%7B2%7D+%26+%5Ctext%7B++%7CY-f%28x%29%7C%3C1%7D++%5C%5C++++%7CY-f%28x%29%7C-%5Cfrac%7B1%7D%7B2%7D+++++++%26+%5Ctext%7B+%7CY-f%28x%29%7C%3E%3D1%7D++%5Cend%7Bcases%7D%5C%5C'>    
在fast-RCNN中提出，主要用于目标检测中防止梯度爆炸  
coding：
```
def SmoothL1loss(x, y):
    assert len(x) == len(y)
    for i_x, i_y in zip(x, y):
        tmp = abs(i_y - i_x)
        if tmp < 1:
            loss += 0.5 * (tmp ** 2)
        else:
            loss = tmp - 0.5
```  
#### 3.1.5 Huber loss
eq: <img src='https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29+%3D++%5Cbegin%7Bcases%7D+++++%5Cfrac%7B1%7D%7B2%7D%28Y-f%28x%29%29%5E%7B2%7D+%26+%5Ctext%7B++%7CY-f%28x%29%7C%7D++%3C%3D+%5Cdelta+%5C%5C++++%5Cdelta%7CY-f%28x%29%7C-%5Cfrac%7B1%7D%7B2%7D++++%5Cdelta%5E%7B2%7D++%26+%5Ctext%7B+%7CY-f%28x%29%7C%7D+%3E+%5Cdelta+%5Cend%7Bcases%7D%5C%5C'>  
