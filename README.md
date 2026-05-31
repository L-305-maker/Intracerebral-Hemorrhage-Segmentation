# Intracerebral Hemorrhage Segmentation
***
## 1、实验意义所在
- 辅助医生快速定位病灶所在
- 自动量化出血体积
- 为病情评估和治疗决策提供依据
- 为医学AI提供可解释性结果
***
## 2、目前挑战所在
- 出血形状不规则，常呈现出斑片状、条带状等，且会出现边界模糊、多灶分布等情况，普通的U-net的分割效果有限
- 小病灶在整张图片中所占像素极少，模型更倾向于预测背景，小出血区域召回率低
- CT图像存在噪声，例如颅骨高密度区域、钙化灶和运动伪影等
- 数据集有限，且标注成本较高
***
## 3、本实验目前的进度
- 将baseline与Advanced_Unet进行对比试验，分辨哪一些模块的加入可以改进模型的性能
- 在Advanced_Unet的内部作对实验：
    - 单卷积层与双卷积层进行对比试验
    - 在基础上再进行，对于不同dropout数值对于结果中Dice、Iou的影响分析
    - 试验BatchNorm层的有无对于实验结果的影响
    - 对不同的Loss函数对于实验结果影响的分析实验
***
## 4、项目结构
```text
Intracerebral Hemorrhage Segmentation
│
├──.gitignore
├──requirements.txt        #依赖库目录
├──main.py                 #项目的统一入口
├──README.md               
│
├──src/
│   ├──criterion/          #不同的损失函数，用于进行效果的对比
│   │   ├──Focal_Loss.py
│   │   ├──Tversky_Loss.py
│   │   └──Weighted_DiceLoss.py
│   │
│   ├──models/             #不同的模型，包括的基础模型以及更加优秀的模型
│   │   ├──baseline.py
│   │   └──Advanced_model.py
│   │
│   ├──args_parse.py       #参数调整模块
│   ├──data_process.py     #对原始数据进行处理
│   ├──result_store.py     #结果存储工具模块
│   ├──train_test.py       #训练-验证-测试模块
│   └──visualize.py        #对结果可视化处理模块
│
├──outputs/                #结果图像存储位置
│
└──configs/                #训练的历史数据存储位置
```
***
## 5、结果展示
### （1）、baseline与Advanced-Unet的对比实验
![ ](outputs\visualizations\compare_01_model.png)
|model_name|Simple-Unet|Advanced-Unet|
|---|---|---|
|Dice|0.7695|0.7831|
|Iou|0.6780|0.6963|
>这里的Advanced-Unet属于最基础版本
从实验结果来看，基础的Advanced-Unet的Dice和Iou都要高出Simple-Unet约2个百分点
### （2）、单卷积层与双卷积层的对比实验
![ ](outputs\visualizations\compare_05_Double_Conv.png)
|Conv2d_number|one|two|
|---|---|---|
|Dice|0.7831|0.7907|
|Iou|0.6780|0.7188|
### （3）、是否有BatchNorm层的对比实验
![ ](outputs\visualizations\compare_04_BatchNorm.png)
|use_BatchNorm|True|False|
|---|---|---|
|Dice|0.8356|0.7907|
|Iou|0.7736|0.7188|
### （4）、不同Loss函数的对比实验
![ ](outputs\visualizations\compare_02_criterion.png)
|Criterion|BCEWithLogitLoss|Focal_Loss|Tversky_Loss|Weighted_DiceLoss|
|---|---|---|---|---|
|Dice|0.8356|0.8495|0.8581|0.8483|
|Iou|0.7736|0.7867|0.7926|0.7911|
### （5）、不同dropout数值的对比实验
![ ](outputs\visualizations\compare_03_dropout.png)
|dropout|0|0.1|0.2|0.3|
|---|---|---|---|---|
|Dice|0.8356|0.8509|0.8424|0.8407|
|Iou|0.7736|0.7903|0.7841|0.7803|
>该实验的损失函数时BCEWithLogitLoss，后续会再做一个最好的组合的实验
### （6）、总对比图像
![ ](outputs\visualizations\all_experiments.png)
### (7)、综合以上实验的结论：
dropout = 0.1，损失函数为Tversky_Loss，同时采用双卷积层和BatchNorm的效果最好，后续也会对这个结果进行进一步的探究
***
## 6、后续提升部分
- 后续会继续引入Attention-Unet等模型
- 现在的数据增强过于简单，后续会继续引入其他数据增强方法：随机噪声、随机缩放、弹性变形和随机亮度/对比度等
- 进行CT窗位窗宽处理
- 对分割结果进行后处理：连通域分析去除小噪点、形态学操作平滑边界、CRF 精修
- 增加更多评估指标：Sensitivity（召回率）、Specificity、Hausdorff Distance