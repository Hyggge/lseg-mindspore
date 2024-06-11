import torch as pytorch_torch # Only use PyTorch for generating input for pytorch model
import msadapter.pytorch as torch
import msadapter.pytorch.nn as nn
# import timm
import types
import math
import msadapter.pytorch.nn.functional as F
import clip
from mindconverter import pytorch2mindspore
from mindspore import Tensor as msTensor
from mindspore import float16 as msFloat16
from collections import OrderedDict # Only use for CLIP Transformer Initialization
import mindcv
from mindspore import context
import traceback
from vit_convert_py2ms.tokenizer import tokenize


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


attention = {}


def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * module.scale

        attn = attn.softmax(dim=-1)  # [:,:,1,1:]
        attention[name] = attn

    return hook


def get_mean_attention_map(attn, token, shape):
    attn = attn[:, :, token, 1:]
    attn = attn.unflatten(2, torch.Size([shape[2] // 16, shape[3] // 16])).float()
    attn = torch.nn.functional.interpolate(
        attn, size=shape[2:], mode="bicubic", align_corners=False
    ).squeeze(0)

    all_attn = torch.mean(attn, 0)

    return all_attn


class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index

        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)

        return self.project(features)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x


def forward_vit(pretrained, x):
    b, c, h, w = x.shape
    
    # encoder
    glob = pretrained.model.forward_flex(x)

    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]

    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // pretrained.model.patch_size[1],
                    w // pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = pretrained.act_postprocess1[3 : len(pretrained.act_postprocess1)](layer_1)
    layer_2 = pretrained.act_postprocess2[3 : len(pretrained.act_postprocess2)](layer_2)
    layer_3 = pretrained.act_postprocess3[3 : len(pretrained.act_postprocess3)](layer_3)
    layer_4 = pretrained.act_postprocess4[3 : len(pretrained.act_postprocess4)](layer_4)

    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x):
    b, c, h, w = x.shape

    pos_embed = self._resize_pos_embed(
        self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
    )

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x


def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

class ResidualAttentionBlock(nn.Module):
    # ResidualAttentionBlock used in CLIP
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    # Transfomer used in CLIP
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MS_CLIP_TextEncoder(nn.Module):
    def __init__(self, torch_clip):
        super(MS_CLIP_TextEncoder, self).__init__()
        self.vocab_size = 49408
        self.transformer_width = 512
        self.transformer_layers = 12
        self.transformer_heads = 8
        self.context_length = 77
        self.embed_dim = 512
        self.token_embedding = nn.Embedding(self.vocab_size, self.transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.transformer_width))
        self.text_projection = nn.Parameter(torch.empty(self.transformer_width, self.embed_dim))
        self.ln_final = LayerNorm(self.transformer_width)


        self.transformer = Transformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        # self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # self.token_embedding = torch_clip.token_embedding
        # self.positional_embedding = torch_clip.positional_embedding
        # self.transformer = torch_clip.transformer
        # self.ln_final = torch_clip.ln_final
        # self.text_projection = torch_clip.text_projection
        self.dtype = msFloat16
        print(f"dtype is {self.dtype}")
        # Initialize parameters-weight from torch-clip
        self.token_embedding.weight = self.weight_convert(torch_clip.token_embedding.weight)
        self.positional_embedding = msTensor(self.weight_convert(torch_clip.positional_embedding), msFloat16)
        self.text_projection = self.weight_convert(torch_clip.text_projection)
        for res_id in range(self.transformer_layers):
            (self.transformer.resblocks)[res_id].attn.in_proj_weight = self.weight_convert((torch_clip.transformer.resblocks)[res_id].attn.in_proj_weight)
            (self.transformer.resblocks)[res_id].attn.out_proj.weight = self.weight_convert((torch_clip.transformer.resblocks)[res_id].attn.out_proj.weight)
            (self.transformer.resblocks)[res_id].mlp.c_fc.weight = self.weight_convert((torch_clip.transformer.resblocks)[res_id].mlp.c_fc.weight)
            (self.transformer.resblocks)[res_id].mlp.c_proj.weight = self.weight_convert((torch_clip.transformer.resblocks)[res_id].mlp.c_proj.weight)

        # Precision Convert
        self.token_embedding.to_float(msFloat16)
        # self.positional_embedding.to_float(msFloat16)

    def weight_convert(self, pytorch_weight):
        # Convert to mindspore weight
        return msTensor(pytorch_weight.cpu().detach().numpy())

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        # MindSpore API
        mask.fill_adapter(float("-inf"))
        mask.triu(1)  # zero out the lower diagonal
        # PyTorch API
        # mask.fill_(float("-inf"))
        # mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text):
        # Code from CLIP @ encode_text for text encoding
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, text):
        # Code from CLIP @ encode_text for pytorch2mindspore convert
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        # print(x)
        # print(x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # PyTorch API
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        # MindSpore API
        x = x[torch.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection

        return x

def _make_pretrained_clip_vitl16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    # Convert PyTorch Model to MindSpore Model
    from PIL import Image
    import os
    import shutil
    ################################################################## MS-CLIP
    # import open_clip
    # OpenAI CLIP
    clip_pretrained, _ = clip.load("ViT-B/32", device='cpu', jit=False)

    # OPenClip-for-MindSpore on Hugging Face
    # clip_pretrained, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    # tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # MindFormers CLIP-Model required Ascend-Env
    # from mindformers import CLIPModel
    # clip_pretrained = CLIPModel.from_pretrained("clip_vit_b_32")

    print("Initialize MindSpore-CLIP")
    text_input = tokenize(["a diagram", "a dog", "a cat"]).to('cpu')
    # image_input = _(Image.open("./inputs/cat1.jpeg")).unsqueeze(0).to('cuda')
    clip_input = msTensor(text_input)
    ms_clip_path = "./clip-convert-py2ms"
    if os.path.exists(ms_clip_path) and os.path.isdir(ms_clip_path):
        shutil.rmtree(ms_clip_path)
    clip_pretrained = MS_CLIP_TextEncoder(clip_pretrained)
    # pytorch2mindspore(clip_pretrained, clip_input, output_dir=ms_clip_path)
    # print(clip_input)

    print("Finish clip initialization")
    ms_clip_output = clip_pretrained(clip_input)
    ################################################################## MS-CLIP



    ################################################################## MS-VIT

    # model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)
    model = mindcv.create_model("vit_l_16_224", pretrained=pretrained) # Only 224 * 224 pretrained model finded
    
    print("Load MindSpore-VIT")


    # Only use PyTorch for generating input for pytorch model
    # In order to satisfy `dummy_input` in API `pytorch2mindspore`
    input_shape = (1, 3, 224, 224)
    model_input = (torch.randn(*input_shape), )
    ms_vit_path = "./vit_convert_py2ms"
    # if os.path.exists(ms_vit_path) and os.path.isdir(ms_vit_path):
    #     shutil.rmtree(ms_vit_path)
    # pytorch2mindspore(model, model_input, output_dir=ms_vit_path)
    # print("Finish vit-model convert")
    # Test ms-vit-model vit_convert_py2ms/model.py
    # from vit_convert_py2ms.model import MindSporeModel as MS_VIT_Model
    # from mindspore import load_checkpoint, load_param_into_net
    # model_param = load_checkpoint(f"{ms_vit_path}/model.ckpt")
    # model = MS_VIT_Model()
    # load_param_into_net(model, model_param)
    ms_vit_input = torch.randn(*input_shape)
    ms_vit_output = model(ms_vit_input)

    ################################################################## MS-VIT

    # Convert PyTorch Model to MindSpore Model

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    
    pretrained = _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained, pretrained


def _make_pretrained_clipRN50x16_vitl16_384(
    pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False
):
    clip_pretrained, _ = clip.load("RN50x16", device='cuda', jit=False)
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained)

    hooks = [5, 11, 17, 23] if hooks == None else hooks
    
    pretrained = _make_vit_b16_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
        enable_attention_hooks=enable_attention_hooks,
    )
    return clip_pretrained, pretrained


def _make_pretrained_clip_vitb32_384(pretrained, use_readout="ignore", hooks=None, enable_attention_hooks=False):
    clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
    model = timm.create_model("vit_base_patch32_384", pretrained=pretrained)

    hooks = [2, 5, 8, 11] if hooks == None else hooks
    
    pretrained = _make_vit_b32_backbone(
        model, 
        features=[96, 192, 384, 768], 
        hooks=hooks, 
        use_readout=use_readout,
        enable_attention_hooks=False,
    )
    return clip_pretrained, pretrained


def _make_vit_b32_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()
    
    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    pretrained.model.patch_size = [32, 32]
    pretrained.model.start_index = start_index

    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // pretrained.model.patch_size[1], size[1] // pretrained.model.patch_size[0]])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=8,
            stride=8,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // pretrained.model.patch_size[1], size[1] // pretrained.model.patch_size[0]])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // pretrained.model.patch_size[1], size[1] // pretrained.model.patch_size[0]])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[2],
            out_channels=features[2],
            kernel_size=2,
            stride=2,
            padding=0,
            # output_padding=output_padding,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // pretrained.model.patch_size[1], size[1] // pretrained.model.patch_size[0]])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )
    
    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_vit_b16_backbone(
    model,
    features=[96, 192, 384, 768],
    size=[384, 384],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    use_readout="ignore",
    start_index=1,
    enable_attention_hooks=False,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    if enable_attention_hooks:
        pretrained.model.blocks[hooks[0]].attn.register_forward_hook(
            get_attention("attn_1")
        )
        pretrained.model.blocks[hooks[1]].attn.register_forward_hook(
            get_attention("attn_2")
        )
        pretrained.model.blocks[hooks[2]].attn.register_forward_hook(
            get_attention("attn_3")
        )
        pretrained.model.blocks[hooks[3]].attn.register_forward_hook(
            get_attention("attn_4")
        )
        pretrained.attention = attention

    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // 16, size[1] // 16])),
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained