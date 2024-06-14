# LSeg Implemented by Mindspore


## 论文介绍

### 背景

语义分割是计算机视觉的核心任务之一，旨在将图像划分成具有明确语义类别标签的各个区域。

大多数语义分割方法，都是先预定义一个固定的语义标签集合，然后学习如何将这些标签分配给每个像素。这种方式有两个局限性：

- 语义标签集合的大小通常由训练数据集决定，范围从数十到数百不等。然而，现实世界中存在成千上万个潜在的类别，这限制了现有模型的识别能力。
- 创建大型且多样化的训练数据集需要大量的人工注释工作，这不仅成本高昂，而且在类别数量增加时，对注释者的细致程度和一致性要求也随之提高。

论文提出的 LSeg 模型通过结合文本嵌入和图像嵌入，可以针对任意长度和顺序的语义标签集合进行分割，不受固定语义标签集合的限制，提高了语义分割的灵活性和通用性。另外，当 LSeg 在一个语义分割数据集上训练后，可以直接泛化到训练时未见过的标签，从而实现 zero-shot。

![image-20240613232604517](./asserts/example.png)

### 论文方法

论文方法的核心思想是：将文本标签和图像像素嵌入同一个空间中，并为每个像素分配“相距最近”的标签。LSeg 整体框架如下图所示：

![image-20240613233703057](./asserts/lseg.png)

LSeg 由 3 个部分组成：Text Encoder、Image Encoder、Spatial  Regularization Block。

#### Text Encoder 和 Image Encoder

Text Encoder 用于将 N 个文本标签嵌入到一个连续的向量空间 $\mathbb{R}^n$中，生成 $N$ 个向量  $T_1,\cdots, T_n \in \mathbb{R}^n$ 。和 Text Encoder 类似，Image Encoder 用于获得经过下采样的图像中每个像素的嵌入向量（为了节省内存），得到一个低分辨率的特征图。假设输入图像大小为 $H \times W$ ， $s$ 是下采样比率，则得到的特征图为 $I \in \mathbb{R}^{\widetilde{H} \times \widetilde{W} \times C}$，其中 $\widetilde{H} = \frac{H}{s}, \widetilde{W} = \frac{W}{s}$。

当得到文本标签和图像像素的嵌入后，可以计算每个像素嵌入和所有标签嵌入的内积，从而得到一个新的张量 $F \in \mathbb{R}^{\widetilde{H} \times \widetilde{W} \times N}$ 。其中， $F_{ijk}$  表示坐标 $(i, j)$ 的像素和第k个标签的内积，即  $F_{ijk} = I_{ij} \cdot T_k$ 。在预测时，将与某个像素嵌入的内积最大的标签作为预测结果。因此，训练目标就是最大化像素嵌入和真实标签嵌入的内积，通过结合 softmax 和交叉熵损失函数即可实现。

#### Spatial  Regularization Block

由于内存限制，图像编码器以较低的分辨率预测像素嵌入，因此得到的 “标签-像素”内积矩阵 $F \in \mathbb{R}^{\widetilde{H} \times \widetilde{W} \times N}$ 分辨率也比较低。在预测/训练时，需要得到分辨率和原图相同的分割结果。因此，需要先对 $F$ 进行后处理，恢复分辨率，得到   $F' \in \mathbb{R}^{H \times W \times N}$ ，然后根据 $F'$ 为每个像素分配标签。Spatial  Regularization Block 就是进行后处理的模块。在  Spatial  Regularization Block 中， $F$ 先通过 Depthwise Block 或者 Bottlenect Block 进行变换，然后通过双线性插值进行上采样就得到了 $F'$  。

Depthwise Block 和 Bottlenect Block 的结构如下所示。

![image-20240614003844751](./asserts/regularization.png)

#### 实现细节

Text Encoder 使用的是预训练的 CLIP 文本编码器，在训练时冻结。Image Encoder 使用的是在 ImageNet 上预训练的 ViT 或者 ResNet，在训练时更新。网络其他参数进行随机初始化。

 

### 数据集

[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)包含27574张图像（训练集25574张，测试集2000张），共365种场景。在这些图像中，一共有来自3688个不同类别的共707868个不同对象。其中，数据标注类别分为三类：

● 离散对象（discrete object），它是具有明确定义的形状的事物，例如汽车、人；

● 包含无定形背景区域的东西（stuff），例如草、天空；

● 对象部分（object part），它是某些具有功能意义的现有对象实例的组件，例如头部或腿部。

![ADE20K](./asserts/ADE20K.png)

## Pytorch 实现版本

官方仓库 [Lang-SEG](https://github.com/isl-org/lang-seg)是基于 Pytorch-Lightning 框架进行搭建。由于存在依赖 Pytorch 的第三方库 ，因此不能直接进行转化为 Mindspore 版本。我们现将原仓库代码转化为纯Pytorch版本，然后再迁移到 Mindspore 框架中。本仓库 torch 分支是使用原生 Pytorch 实现的代码。

### 实现思路

Pytorch-Lightning 只是对 dataloader的创建、损失函数的创建、单步训练/验证的流程进行了封装，LSeg 网络的构建使用的是仍然是原生 Pytorch。我们只需要参考官网"[How to Organize PyTorch Into Lightning](https://lightning.ai/docs/pytorch/stable/starter/converting.html#use-your-own-data)"，对该过程进行逆向处理，就可以得到原生Pytorch 版本的训练框架。具体流程如下

- Step1:  从 pytorch_lightning.LightningModule 模块中找到 dataloader 创建函数（train_dataloader和val_dataloader）、优化器创建函数（configure_optimizers），将其直接迁移过来，封装成新的函数 get_train_dataloader, get_val_dataloader, get_optimizer，便于后续使用。

    ```python
    def get_train_dataloader(...):
        # content of `train_dataloader` function in `pytorch_lightning.LightningModule`
    
    def get_val_dataloader(...):
        # content of `val_dataloader` function in `pytorch_lightning.LightningModule`
    
    def get_optimizer(...):
        # content of `configure_optimizers` function in `pytorch_lightning.LightningModule`

- Step 2:  从 pytorch_lightning.LightningModule 模块中找到定义“单步训练/验证流程“ 的函数（training_step和validation_step），根据原生 pytorch 训练/验证方式，得到每个训练/验证 epoch 的流程，封装成 train、val 两个函数。

    ```python
    def train(...):
        model.train()
        sample_num = 0
        train_loss_total = 0
        loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
    
        for i, batch in loop:
            # get input data
            img, target = batch
            # forward
            # content of `training_step` function in `pytorch_lightning.LightningModule`
            out = model(img)
            loss = criterion(out, target)
            # backward
            loss.backward()
            # update metric
            sample_num += 1
            train_loss_total += loss.item()
            # Show progress while training
            loop.set_description(f'Epoch {epoch + 1}')
            loop.set_postfix(batch_loss=loss.item())
    
        return {
            "train_loss_avg": train_loss_total / sample_num
        }
    
    def val(...):
        sample_num = 0
        val_loss_total = 0
        metric.reset()
        loop= tqdm(enumerate(dataloader), leave=False, total=len(dataloader)
    
        with torch.no_grad():
            for i, batch in loop:
                # get input data
                img, target = batch
                # forward
                # content of `validation_step` function in `pytorch_lightning.LightningModule`
                out = model(img)
                # update metric
                sample_num += 1
                val_loss_total += val_loss.item()
                metric.update(target, final_output)
                # Show progress while evalating
                loop.set_description(f'Evalating')
    
        pixAcc, iou = metric.get()
        return {
            "pixAcc": pixAcc,
            "iou": iou,
            "val_loss_avg": val_loss_total / sample_num
        }
    ```
    
- step3：在主函数中，先解析命令行得到配置参数，然后构建 dataloader、损失函数、模型、优化器、评价指标，最后开始训练流程。每训练一个 epoch，保存对应的checkpoint，并进行一轮评估。

    ```python
    if __name__ == "__main__":
    	args = Options().parser.parse_args()
        train_dataloader = get_train_dataloader()
        val_dataloader = get_val_dataloader()
        criterion = SegmentationLosses()
        model = LSegNet()
        optimizer, scheduler = get_optimizer()
        metric = SegmentationMetric(len(labels))
        # train start
        for epoch in range(args.max_epochs):
            # train
            train_result_epoch = train(...)
            print(f"Train loss: {train_result_epoch['train_loss_avg']:.4f}")
            # save checkpoint
            ckpt_path = os.path.join(log_dir, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            # validate
            val_result_epoch = val(...)
            print(f"Validation loss: {val_result_epoch['val_loss_avg']:.4f}, \
                    pixAcc: {val_result_epoch['pixAcc']:.4f}, \
                    iou: {val_result_epoch['iou']:.4f}")
    ```

为了便于展示，上面展示的代码都进行了简化。详细代码可见 train_custom.py 文件。

### 环境准备

1. 使用 conda 创建 python 环境，推荐 python 版本为3.8

```
conda create -n lseg python=3.8
```


2.  安装 torch 和其他依赖项

```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html 
pip install -r pip install -r requirements.txt
```

3. 数据集准备：本仓库选用[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)进行训练及测试，其数据下载及准备方式如下

```
python prepare_ade20k.py
```

### 运行命令

执行 `train_custom.sh` 即可开始训练。另外可以通过修改脚本命令行参数来调整实验设置，`options.py` 文件中展示了所有可用参数。

```
sh train_custom.sh
```



## Mindspore 实现版本

本仓库的代码基于官方仓库[Lang-SEG](https://github.com/isl-org/lang-seg)进行二次开发，主要使用[MSAdapter](https://openi.pcl.ac.cn/OpenI/MSAdapter)、[MindCV](https://mindspore-lab.github.io/mindcv/zh/)、[MindConverter](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.7/migrate_3rd_scripts_mindconverter.html)、[MindFormers](https://gitee.com/mindspore/mindformers)进行模型转化、网络迁移与实现。

### 网络迁移

由于[Lang-SEG](https://github.com/isl-org/lang-seg)的网络结构基于PyTorch版本的[VIT](https://github.com/google-research/vision_transformer)与[CLIP](https://github.com/openai/CLIP)进行实现，所以本仓库首先使用[MindCV](https://mindspore-lab.github.io/mindcv/zh/)、[MindConverter](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.7/migrate_3rd_scripts_mindconverter.html)与[MindFormers](https://gitee.com/mindspore/mindformers)进行模型转化，获取相应的MindSpore版本的VIT与CLIP。其中，MindCV是一个基于MindSpore开发的，致力于计算机视觉相关技术研发的开源工具箱。它提供了大量的计算机视觉领域的经典模型和SoTA模型以及它们的预训练权重和训练策略。MindConverter则是一款模型迁移工具，可将PyTorch(ONNX)或Tensorflow(PB)模型快速迁移到MindSpore框架下使用。模型文件（ONNX/PB）包含网络模型结构（`network`）与权重信息（`weights`），迁移后将生成MindSpore框架下的模型定义脚本（`model.py`）与权重文件（`ckpt`）。其使用示例如下：

**VIT**

```python
# timm in PyTorch：获取PyTorch VIT
import timm
vit_model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

# mindcv in MindSpore：获取MindSpore VIT
import mindcv
vit_model = mindcv.create_model("vit_l_16_224", pretrained=pretrained)

```

**CLIP**

方式1：基于[MindConverter](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.7/migrate_3rd_scripts_mindconverter.html)与[PyTorch-CLIP](https://github.com/openai/CLIP)使用pytorch2mindspore接口进行模型转换。

```python
# 导入PyTorch CLIP
import clip
pytorch_clip_pretrained, _ = clip.load("ViT-B/32", device='cpu')

# 转化为MindSpore CLIP
from mindconverter import pytorch2mindspore
ms_clip_path = "./clip-convert-py2ms"
pytorch2mindspore(pytorch_clip_pretrained, clip_input, output_dir=ms_clip_path)
```

方式2：在Ascend环境下通过[MindFormers](https://gitee.com/mindspore/mindformers)获取MindSpore-CLIP模型。

```python
from mindformers import CLIPModel
clip_pretrained = CLIPModel.from_pretrained("clip_vit_b_32")
```

方式3：通过Hugging Face获取MindSpore-CLIP模型。

```python
import open_clip
clip_pretrained, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
```

方式4：基于MindSpore实现CLIP并对[PyTorch-CLIP](https://github.com/openai/CLIP)进行参数转换。

```python
import clip
pytorch_clip_pretrained, _ = clip.load("ViT-B/32", device='cpu')
clip_input = clip.tokenize(["a diagram", "a dog", "a cat"]).to('cpu')
ms_clip_path = "./clip-convert-py2ms"
# MS_CLIP_TextEncoder为基于MindSpore实现的编码器
clip_pretrained = MS_CLIP_TextEncoder(pytorch_clip_pretrained)
pytorch2mindspore(clip_pretrained, clip_input, output_dir=ms_clip_path)
```

最后，本仓库在进行模型对接、整合及实现LSegNet时使用了[MSAdapter](https://openi.pcl.ac.cn/OpenI/MSAdapter)工具。该工具以PyTorch接口为标准，为用户提供了一套与PyTorch一样（接口和功能完全一致）的中高阶模型构建接口和数据处理接口。其使用示例如下：

```python
# import torch
# import torch.nn as nn
# import torchvision import datasets, transforms

import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
import msadapter.pytorch.nn.functional as F
import msadapter.torchvision import datasets, transforms
```

其中，由于部分MindSpore接口与PyTorch接口不适配，本仓库对其进行了进一步转化，此处以`transpose`与`expand`为例：

```python
def transpose_custom(tensor, dim0: int, dim1: int):
    # Ref to https://www.mindspore.cn/docs/zh-CN/r1.9/note/api_mapping/pytorch_diff/Tensor.transpose.html
    size = len(tensor.shape)
    dims = [i for i in range(size)]
    dims[dim0], dims[dim1] = dim1, dim0
    dims = tuple(dims)
    return tensor.transpose(dims)
```

```python
# Ref to https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_diff/expand.html
# PyTorch
import torch
x = torch.tensor([1, 2, 3])
output = x.expand(3, 3)

# MindSpore
import mindspore
import numpy as np
from mindspore import Tensor
shape = (3, 3)
x = Tensor(np.array([1, 2, 3]))
output = x.broadcast_to(shape)
```

### 训练框架迁移

之前得到了原生 pytorch 实现的训练框架，接下来只需要使用 MSAdapter 将 dataloader、优化器、损失函数、评价指标、单步训练/验证流程迁移到 Mindspore 框架即可。 在迁移时，只对 `train_custom.py` 进行修改。

**更换依赖**

首先，删除 train_custom.py 中 torch 相关依赖包，引入 MSAdapter 和 Mindspore。

```python
import mindspore as ms
import mindspore.nn as nn
import msadapter.pytorch as torch
import msadapter.torchvision.transforms as transforms
```

**迁移 dataloader**

dataloader 相关的构造函数可以直接被 MSAdapter 迁移到 Mindspore框架下，无需修改 train_custom.py 中相关部分 。但是，数据集定义在  encoding_custom/datasets 中，并且是用 pytorch 实现的。我们只需要将该文件夹下 ade20k.py 和 base.py中 import 部分进行替换即可。以 encoding_custom/datasets/ade20k.py 为例：

``` py
# 修改前
import torch
import torch.utils.data as data
import torchvision.transforms as transform

# 修改后
import msadapter.pytorch as torch
import msadapter.pytorch.utils.data as data
import msadapter.torchvision.transforms as transform
```

**迁移优化器&学习率调度器**

优化器不能被 MSAdapter 直接迁移，需要使用 mindspore 中的 API 重新进行构建。原生 pytorch 版本使用的是带动量的 SGD，使用 `torch.optim.SDG`进行构建；迁移到 Mindspore时，需要使用 `mindspore.nn.Momentum `进行构建。

在迁移学习率调度器时，需要注意两点：

- 在pytorch 中，优化器先创建，然后作为参数传入学习率调度器的构造函数中。而在 Mindspore 中，学习率调度器先创建，并作为参数传入优化器的构造函数中的 learning_rate 形参位置。

- 在原生pytorch实现的训练框架中，学习率调度器使用的是 torch.optim.lr_scheduler.LambdaLR，可以通过传入lamba表达式自定义学习率调整规则。MindSpore未提供类似的lambda接口，只能通过自定义函数或自定义 `LearningRateSchedule` 来实现。官方文档中有对应例子：

  ![image-20240613205948467](./asserts/lambda_lr.png) 

最后迁移前后的代码如下所示：

```python
# 迁移前
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=base_lr,
    momentum=0.9,
    weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda x: pow(1.0 - x / max_epochs, 0.9)
)

# 迁移后
class DynamicDecayLR(nn.learning_rate_schedule.LearningRateSchedule):
    def __init__(self, lr, step_per_epoch, max_epochs):
        super(DynamicDecayLR, self).__init__()
        self.lr = lr
        self.step_per_epoch = step_per_epoch
        self.max_epochs = max_epochs

    def construct(self, global_step):
        current_epoch = global_step // self.step_per_epoch
        return self.lr * pow(1.0 - current_epoch / self.max_epochs, 0.9)

opt = nn.Momentum(
    model.parameters(),
    learning_rate=DynamicDecayLR(base_lr, step_per_epoch, max_epochs),
    momentum=0.9,
    weight_decay=weight_decay,
)

```

**迁移损失函数&评价指标**

在原生 pytorch 实现的训练流程中，损失函数和评价指标使用的是 encoding_custom文件夹中定义的 SegmentationLosses 和  SegmentationMetric。这两个类都是使用 pytorch 实现的。我们只需要将其所在文件的 import 部分进行替换即可。

**迁移单步训练/验证流程**

mindspore 中定义了用于单步训练/验证的中低层API `mindspore.nn.TrainOneStepCell` 和 `mindspore.nn.WithEvalCell`，在创建时传入模型和损失函数作为实例属性，调用时只需要传入图像和GT即可返回loss，同时进行更新。

迁移后的单epoch训练/验证函数如下所示（简化版）

```py
def train(...):
    model.train()
    sample_num = 0
    train_loss_total = 0
    loop = tqdm(enumerate(dataloader), leave=False, total=len(dataloader))
	model_with_loss = nn.WithLossCell(model, criterion)
    train_step = nn.TrainOneStepCell(model_with_loss, optimizer)
    
    for i, batch in loop:
        # get input data
        img, target = batch
        # forward & backward
        loss = train_step(img, target)
        # update metric
        sample_num += 1
        train_loss_total += loss.item()
        # Show progress while training
        loop.set_description(f'Epoch {epoch + 1}')
        loop.set_postfix(batch_loss=loss.item())

    return {
        "train_loss_avg": train_loss_total / sample_num
    }

def val(...):
    sample_num = 0
    val_loss_total = 0
    metric.reset()
    loop= tqdm(enumerate(dataloader), leave=False, total=len(dataloader)
	model_with_eval = nn.WithEvalCell(model, criterion)
    with torch.no_grad():
        for i, batch in loop:
            # get input data
            img, target = batch
            # forward
            val_loss, out, target = model_with_eval(img, target)
            # update metric
            sample_num += 1
            val_loss_total += val_loss.item()
            metric.update(target, final_output)
            # Show progress while evalating
            loop.set_description(f'Evalating')

    pixAcc, iou = metric.get()
    return {
        "pixAcc": pixAcc,
        "iou": iou,
        "val_loss_avg": val_loss_total / sample_num
    }
```

**其他细节**

1. 在 pytorch 中，需要使用 `to` 函数将模型和张量手动 “搬运” 到GPU显存上，而在 mindspore 中，只需要通过 `ms.set_context(device_target='GPU')` 一行代码即可指定运行设备为GPU，无需手动“搬运”。
2. pytorch 使用 `torch.save`函数来保存模型，mindspore 中对应 API 的为 `mindspore.save_checkpoint`。

### 环境准备

1. 使用 conda 创建 python 环境，推荐 python 版本为3.8

```
conda create -n lseg  python=3.8
```

2.  安装 mindspore相关库和其他依赖项：对于LSeg的基本环境，本仓库沿用官方仓库[Lang-SEG](https://github.com/isl-org/lang-seg)的基本配置，为进一步迁移至MindSpore环境，本仓库使用MindSpore相关库，其具体环境及版本号如下：

```
pip install mindspore==2.2.14
pip install msadapter==0.1.0
pip install mindcv==0.3.0
pip install mindconverter==1.7.0
pip install -r pip install -r requirements.txt

# CLIP Env (根据环境不同，可选用不同的CLIP实现方式，下述环境配置为可选项):
pip install open_clip_torch (Hugging Face required)
pip install mindformers (Ascend-Env required)
```

3. 数据集准备：本仓库选用[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)进行训练及测试，其数据下载及准备方式如下

```
python prepare_ade20k.py
```

4. 预训练参数文件准备：本仓库中使用的基于 PyTorch 实现的 CLIP 模型来自[PyTorch-CLIP](https://github.com/openai/CLIP)，本仓库参考了[MindSpore-CLIP](https://github.com/XixinYang/CLIP-mindspore/tree/main)将其转换为基于 MindSpore 的 CLIP 模型，这里提供训练中所使用的[参数文件](https://bhpan.buaa.edu.cn/link/AA15887DBCFC0F477EB523F2CC7814D976)。

### 运行命令

执行 `train_custom.sh` 即可开始训练。另外可以通过修改脚本命令行参数来调整实验设置，`options.py` 文件中展示了所有可用参数。

```
sh train_custom.sh
```



### 训练结果

这里展示了47个Epoch的训练结果：

![image-20240613173528670](./asserts/expriment.png)

| Model          | pixAcc | mIoU  |
| -------------- | ------ | ----- |
| LSeg-PyTorch   | 82.46  | 46.28 |
| LSeg-MindSpore | 81.79  | 47.59 |

> 注：
>
> 1.考虑到原论文训练轮次为240epoch，耗时较长（约2~3h/epoch），本仓库未进行完全复现。
>
> 2.本仓库在迁移部分PyTorch接口及设置至MindSpore框架时进行了微小调整。
