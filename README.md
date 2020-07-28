# Pytorch-NetworkDesign-ShuffleNet-ResNet-MobileNet-BottleNeck

有关单类单目标物体的侦测细节部分，可以跳转我的另一篇博文[DeepLearing—CV系列（二）——单类单目标物体的侦测（框小黄人）与机器学习的评价指标的Pytorch实现](https://blog.csdn.net/wa1tzy/article/details/106805197)查看，**本篇博文针对其网络设计部分与内存溢出问题进行设计。**

增加了对网络设计部分如：瓶颈结构Bottleneck，MobilenetV1,V2，Shufflenet（通道混洗，自定义通道数目混洗），Resnet。

先来看下做出来的效果：matplotlib动态显示

![image](https://img-blog.csdnimg.cn/20200618124744218.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhMXR6eQ==,size_16,color_FFFFFF,t_70)

![image](https://img-blog.csdnimg.cn/20200618124832861.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhMXR6eQ==,size_16,color_FFFFFF,t_70)

![image](https://img-blog.csdnimg.cn/20200618124845605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dhMXR6eQ==,size_16,color_FFFFFF,t_70)

**代码目录**

![image](https://img-blog.csdnimg.cn/20200618130410696.png)

weights:

链接：https://pan.baidu.com/s/12yphOwAipsIlH9PVkbcz3Q 

提取码：sofm 
