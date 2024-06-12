# Language-driven Semantic Segmentation (LSeg)
The repo contains MindSpore Implementation of ICLR'22 paper [Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546).


### Overview


LSeg is a novel model for language-driven semantic image segmentation. LSeg uses a text encoder to compute embeddings of descriptive input labels (e.g., ''grass'' or 'building'') together with a transformer-based image encoder that computes dense per-pixel embeddings of the input image. The image encoder is trained with a contrastive objective to align pixel embeddings to the text embedding of the corresponding semantic class. The text embeddings provide a flexible label representation in which semantically similar labels map to similar regions in the embedding space (e.g., ''cat'' and ''furry''). This allows LSeg to generalize to previously unseen categories at test time, without retraining or even requiring a single additional training sample. We demonstrate that our approach achieves highly competitive zero-shot performance compared to existing zero- and few-shot semantic segmentation methods, and even matches the accuracy of traditional segmentation algorithms when a fixed label set is provided. 

## Usage
### Installation
Envs: 

``` pip install -r requirements.txt ```

Convert PyTorch to MindSpore:
```
MindSpore Env:
  pip install mindspore==2.2.14
  pip install msadapter==0.1.0
  pip install mindcv==0.3.0
  pip install mindconverter==1.7.0
CLIP Env (Optional):
  pip install open_clip_torch (Hugging Face required)
  pip install mindformers (Ascend-Env required)
```

### Data Preparation
By default, for training, testing, we use [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

```
python prepare_ade20k.py
unzip ../datasets/ADEChallengeData2016.zip
```

Download OpenAI's PyTorch-CLIP Model from [PyTorch-CLIP](https://github.com/openai/CLIP) and convert it to MindSpore-CLIP Model refer to  [MindSpore-CLIP](https://github.com/XixinYang/CLIP-mindspore/tree/main). ([CKPTs used in this repo](https://bhpan.buaa.edu.cn/link/AA15887DBCFC0F477EB523F2CC7814D976))

### Training and Testing Example
Training: Backbone = ViT-L/16, Text Encoder from CLIP ViT-B/32

``` bash train_custom.sh ```

Testing: Backbone = ViT-L/16, Text Encoder from CLIP ViT-B/32

``` bash test.sh ```

## Acknowledgement
Thanks to the code base from [DPT](https://github.com/isl-org/DPT), [Pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning), [CLIP](https://github.com/openai/CLIP), [Pytorch Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [Streamlit](https://streamlit.io/), [Wandb](https://wandb.ai/site),  [MindSpore-CLIP](https://github.com/XixinYang/CLIP-mindspore/tree/main), [Lang-SEG](https://github.com/isl-org/lang-seg)

