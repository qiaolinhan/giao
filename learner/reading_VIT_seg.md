# [Reading VIT for Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Strudel_Segmenter_Transformer_for_Semantic_Segmentation_ICCV_2021_paper.pdf)
### Segmenter: Transformer for Semantic Segmentation
```
context 语境； occlusion 遮挡； moderate 缓和； meanwhile 同时； circumvent 规避； leverage 利用； 
```
* 模型: Segmenter.  ViT extend to semantic segmentation  
* 优点： 可以fin-tune 于moderate sized datasets。  
    线性Decoder --> 获得较好结果。 mask transformer generating class masks --> 进一步提升表现。（further improved）  
    Conduct an extensive ablation study （进行了广泛的消融研究）
* 当今现状， challenges： due to rich intra-class variation, context variation and ambiguities originating from occlusions and low image resolution. (遮挡， 低分辨率 --> 类内变化， 上下文变化， 歧义)
* Recent approaches： 依赖卷积编码解码architectures， 
    State-of-art methods： 依赖 learnable stacked convolutions that can capture semantically rich information.  
* 当今现状， 局限性： the access to 全局信息 in image.  --> 规避这个问题 --> feature aggregation with dilated convolutions and spatial pyramaid pooling. (DeepLab) --> spatial attention, point-wise attention to better capture contextual information.  
    但是， 仍然依赖于convolutional backbones --> biased towards local interactions

* 解决： to overcome the limitations, we formulate the problem of sematic segmentation as a sequence-to-sequence problem and use a transformer architecture to leverage contextual information at every stage of the model.


