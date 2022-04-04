学习一下这个loss function。 因为发现segmentation中，自己造的light U-net使用CEloss要么loss贼大，要么loss不下降。找不到原因，神烦。  
ok doge  
## 1. What is Loss Function
用来度量模型预测值，predection， 和真实值， label， 的差异程度的运算函数。  
* 是一个非负实函数。 通常表示为 $L(Y, f(x))$  
* 损失函数越小，模型鲁棒性越好
## 2. Some Commonly Used Loss Functions
### 2.1 基于距离的损失函数
输入数据映射到-基于距离度量的特征空间-（欧氏空间， 汉明空间， 等）  
映射后的样本看作空间上的点， 采用合适的损失函数度量特征空间上-样本真值-和-模型预测值-之间的距离。  
特征空间上两点距离越小-模型预测性能越好
#### 2.1.1 Mean square error loss (MSE Loss)
eq: $$L(Y|f(x)) = \frac{1}{n} \sum_{i = 1}^{N}{(Y_i - f(x_i))^2}$$  <img src='https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%28Y_%7Bi%7D-f%28x_%7Bi%7D%29%29%5E%7B2%7D%7D'>  
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
eq: $$L(Y|f(x)) = \sqrt{\frac{1}{n}\sum_{i = 1}^{N}{(Y_i - f(x_i))^2}}$$   <img src='https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29%3D%5Csqrt%7B%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%28Y_%7Bi%7D-f%28x_%7Bi%7D%29%29%5E%7B2%7D%7D%7D'>   
又称为-欧氏距离。 常用的距离度量方法，通常度量数据点之间的相似程度。  
优点：凸性， 可微性， 独立同分布且高斯噪声下-能提供最大似然估计。
缺点：收敛速度比L1慢，因为梯度会随着预测值接近真实值而不断减小。 对异常数据比L1敏感，这是平方项引起的，异常数据会引起很大的损失。    
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
eq: $$L(Y|f(x)) = \sum_{i = 1}^{N}{|Y_i - f(x_i)|}$$  <img src='https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29%3D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7B%7CY_%7Bi%7D-f%28x_%7Bi%7D%29%7C%7D'>  
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
Huber loss 是平方损失-和-绝对损失-的综合。
advantage： 克服了平方损失和绝对损失的缺点， 损失函数连续可导， 利用MSE梯度随误差减小的特性-可取的更精确的最小值， 对异常点更鲁棒。
disadvantage： 引入了额外参数， 选择合适的参数比较困难，增加了训练和调试的工作量。
coding：  
```
delta = 1.0
def HuberLoss(x, y):
    assert len(x) == len(y)
    loss = 0
    for i_x, i_y in zip(x, y)；
        tmp = abs(i_y - i_x)
        if tmp <= delta:
            loss += 0.5 * (tmp ** 2)
        else:
            loss += tmp * delta - 0.5 * delta ** 2
    return loss
```
### 基于概率分布-度量的损失函数
将样本相似性-转化为-随机事件出现的可能性， 即 通过度量-样本的真实分布-与-估计的分布-之间的距离-判断两者的相似度。 一般用于涉及-概率分布 或 预测类别出现的概率 的应用问题中， 分类问题中-尤其常用。  
#### KL 散度函数 （相对熵）
eq： <img src = 'https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7BY_%7Bi%7D%5Ctimes+log%28%5Cfrac%7BY_%7Bi%7D%7D%7Bf%28x_%7Bi%7D%29%7D%29%7D+'>  
Kullback-Leibler divergence (KL), 也被称为 相对熵。 非对称度量法。 常用于度量-两个概率分布之间的距离。 两个随机分布的相似度越高， KL散度越小。 当 两个随机分布 差别增大 -> KL散度增大 -> 可用于比较 文本标签 或 图像 的相似性。  
演化：  JS散度函数-JS距离， 用于衡量两个概率分布之间的相似度， -> 消除了KL散度非对称问题， 使得-相似度判别更准确。  
coding：  
```
def KLLoss(y_true: list, y_pred: list):
    assert len(y_true) == len(y_pred)
    KL = 0
    for y, fx in zip(y_true, y_pred)
    KL = 0
    for y, fx in zip(y_true, y_pred):
        KL += y * np.log(y/fx)
    return KL
```
#### 3.2.2 Cross-Entropy
eq： <img src='https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29%3D-%5Csum_%7Bi%3D1%7D%5E%7BN%7D%7BY_%7Bi%7Dlog+f%28x_%7Bi%7D%29%7D'>  
信息论中的概念。 最初用于估计平均编码的长度。 引入机器学习后，用于-评估-当前训练得到的概率分布-与真实分布-的差异情况。  
为了将线性 -> 转为 非线性逼近， 提高模型-预测精度 -> tanh, sigmoid, softmax, ReLU.  
CELoss 刻画 实际输出概率-与-期望输出概率-的相似程度。交叉熵值越小， 两个概率分布越接近。  
优点： 有效避免-梯度消散。   
二分类中， 也叫-对数损失函数。  
coding：  
```
def CrossentropyLoss(y_true: list, y_pred: list):
    assert len(y_true) == len(y_pred)
    loss = 0
    for y, fx in zip(y_true, y_pred):
        loss += -y * np.log(fx)
    return loss
```  
当正负样本不均衡， 通常在交叉熵损失函数类别前加参数$\alpha$ <img src='https://www.zhihu.com/equation?tex=CE+%3D++%5Cbegin%7Bcases%7D+++++-%5Calpha+log%28p%29+%26+%5Ctext%7B+++y+%3D+1%7D++%5C%5C++++-%281-%5Calpha+%29log%281-p%29+++++++%26+%5Ctext%7B+y+%3D+0%7D++%5Cend%7Bcases%7D%5C%5C'>  
#### 2.1.3 Softmax 损失函数
eq: <img src = 'https://www.zhihu.com/equation?tex=L%28Y%7Cf%28x%29%29%3D-%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Blog%5Cfrac%7Be%5E%7Bf_%7BY_%7Bi%7D%7D%7D%7D%7B%5Csum_%7Bj%3D1%7D%5E%7Bc%7D%7Be%5E%7Bf_%7Bj%7D%7D%7D%7D%7D'>  
监督学习中广泛应用。  
本质上为逻辑回归模型在多分类任务上的延申， 常作为CNN模型的损失函数。  
本质： 将一个 k维 任意 实向量x 映射成 另一个 k维 实数向量。 其中 输出向量 每个元素 取值范围 都是（0， 1）。  
优点： 类间可分性。  类间距离 的 优化效果 好。  
不足： 类内距离 的优化效果 较差。  
广泛应用于： 分类， 分割， 人脸识别， 图像自动标注， 人脸验证。解决特征分离问题。 但是 softmax学习奥的特征 不具有 足够 的区分性， 因此 常与 对比损失 或 中心损失 组合使用， 以增强区分能力。 
coding：  
```
def SoftmaxLoss(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s

softmax_fc = torch.nn.Softmax(x)
output = softmax_fc(x) 
```
#### Focal Loss
Focal loss 主要 为了解决 难易样本不均衡 的问题， 注意有区别于 正负样本不均衡 的问题。

|-----| 难 | 易 |
|-----|-----|-----|
| 正| 正难| 正易|
| 负| 负难| 负易|  

易分样本虽然 损失很低，但 数量太多， 对模型 效果的提升 贡献很小， 模型 应该重点关注 难分样本， 因此 需要 把 置信高的损失 再降低一些。  
eq: <img src = 'https://www.zhihu.com/equation?tex=FE+%3D++%5Cbegin%7Bcases%7D+++++-%5Calpha%281-p%29%5E%7B%5Cgamma%7D+log%28p%29+%26+%5Ctext%7B+++y+%3D+1%7D++%5C%5C++++-%281-%5Calpha+%29p%5E%7B%5Cgamma%7D+log%281-p%29+++++++%26+%5Ctext%7B+y+%3D+0%7D++%5Cend%7Bcases%7D%5C%5C'>  

