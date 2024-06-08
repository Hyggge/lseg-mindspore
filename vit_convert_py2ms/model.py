import numpy as np
import mindspore
import mindspore.numpy as ms_np
import mindspore.ops as P
from mindspore import nn
from mindspore import Tensor, Parameter


class GELU(nn.Cell):

    def __init__(self):
        super().__init__()
        self.erf = P.Erf()
        self.sqrt = P.Sqrt()
        self.const0 = Tensor(0.5, mindspore.float32)
        self.const1 = Tensor(1.0, mindspore.float32)
        self.const2 = Tensor(2.0, mindspore.float32)

    def construct(self, x):
        return x * self.const0 * (self.const1 + self.erf(x / self.sqrt(self.const2)))


class PatchEmbed(nn.Cell):

    def __init__(self):
        super(PatchEmbed, self).__init__()
        self.conv2d_0 = nn.Conv2d(kernel_size=(16, 16),
                                  in_channels=3,
                                  out_channels=1024,
                                  stride=(16, 16),
                                  dilation=(1, 1),
                                  padding=0,
                                  pad_mode='valid',
                                  group=1,
                                  has_bias=True)
        self.reshape_1 = P.Reshape()
        self.reshape_1_shape = tuple((1, 1024, 576))
        self.transpose_2_input_perm = (0, 2, 1)

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_reshape_1 = self.reshape_1(opt_conv2d_0, self.reshape_1_shape)
        opt_transpose_2 = P.Transpose()(opt_reshape_1, self.transpose_2_input_perm)
        return opt_transpose_2


class Attention(nn.Cell):

    def __init__(self):
        super(Attention, self).__init__()
        self.linear_3 = nn.Dense()
        self.floordiv_4_input_1 = 16
        self.reshape_5_shape2 = 3
        self.reshape_5_shape3 = 16
        self.transpose_6_dims = (2, 0, 3, 1, 4)
        self.transpose_10_input_perm = (0, 1, 3, 2)
        self.mul_12_input_1 = 0.125
        self.softmax_13 = nn.Softmax(axis=-1)
        self.dropout_14 = nn.Dropout()
        self.transpose_16_input_perm = (0, 2, 1, 3)
        self.linear_18 = nn.Dense()
        self.dropout_19 = nn.Dropout()

    def construct(self, x):
        opt_shape_0 = P.Shape()(x)[0]
        opt_shape_1 = P.Shape()(x)[1]
        opt_shape_2 = P.Shape()(x)[2]
        opt_linear_3 = self.linear_3(x)
        opt_floordiv_4 = P.FloorDiv()(opt_shape_2, self.floordiv_4_input_1)
        opt_reshape_5 = opt_linear_3.reshape(opt_shape_0, opt_shape_1, self.reshape_5_shape2, self.reshape_5_shape3,
                                             opt_floordiv_4)
        opt_transpose_6 = opt_reshape_5.transpose(*self.transpose_6_dims)
        opt_primop_7 = opt_transpose_6[0, :, :, :, :]
        opt_primop_8 = opt_transpose_6[1, :, :, :, :]
        opt_primop_9 = opt_transpose_6[2, :, :, :, :]
        opt_transpose_10 = P.Transpose()(opt_primop_8, self.transpose_10_input_perm)
        opt_matmul_11 = P.matmul(opt_primop_7, opt_transpose_10)
        opt_mul_12 = opt_matmul_11 * self.mul_12_input_1
        opt_softmax_13 = self.softmax_13(opt_mul_12)
        opt_dropout_14 = self.dropout_14(opt_softmax_13)
        opt_matmul_15 = P.matmul(opt_dropout_14, opt_primop_9)
        opt_transpose_16 = P.Transpose()(opt_matmul_15, self.transpose_16_input_perm)
        opt_reshape_17 = opt_transpose_16.reshape(opt_shape_0, opt_shape_1, opt_shape_2)
        opt_linear_18 = self.linear_18(opt_reshape_17)
        opt_dropout_19 = self.dropout_19(opt_linear_18)
        return opt_dropout_19


class Mlp(nn.Cell):

    def __init__(self):
        super(Mlp, self).__init__()
        self.linear_0 = nn.Dense()
        self.gelu_1 = GELU()
        self.dropout_2 = nn.Dropout()
        self.linear_3 = nn.Dense()
        self.dropout_4 = nn.Dropout()

    def construct(self, x):
        opt_linear_0 = self.linear_0(x)
        opt_gelu_1 = self.gelu_1(opt_linear_0)
        opt_dropout_2 = self.dropout_2(opt_gelu_1)
        opt_linear_3 = self.linear_3(opt_dropout_2)
        opt_dropout_4 = self.dropout_4(opt_linear_3)
        return opt_dropout_4


class Block(nn.Cell):

    def __init__(self):
        super(Block, self).__init__()
        self.layernorm_0 = nn.LayerNorm(normalized_shape=(1024, ), epsilon=1e-06)
        self.attention_0 = Attention()
        self.add_1_alpha = 1
        self.layernorm_2 = nn.LayerNorm(normalized_shape=(1024, ), epsilon=1e-06)
        self.mlp_0 = Mlp()
        self.add_3_alpha = 1

    def construct(self, x):
        opt_layernorm_0 = self.layernorm_0(x)
        attention_0_opt = self.attention_0(opt_layernorm_0)
        opt_add_1 = x + attention_0_opt
        opt_layernorm_2 = self.layernorm_2(opt_add_1)
        mlp_0_opt = self.mlp_0(opt_layernorm_2)
        opt_add_3 = opt_add_1 + mlp_0_opt
        return opt_add_3

    # assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    # x = nn.LayerNorm(dtype=self.dtype)(inputs)
    # x = nn.MultiHeadDotProductAttention(
    #     dtype=self.dtype,
    #     kernel_init=nn.initializers.xavier_uniform(),
    #     broadcast_dropout=False,
    #     deterministic=deterministic,
    #     dropout_rate=self.attention_dropout_rate,
    #     num_heads=self.num_heads)(
    #         x, x)
    # x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    # x = x + inputs

    # # MLP block.
    # y = nn.LayerNorm(dtype=self.dtype)(x)
    # y = MlpBlock(
    #     mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate)(
    #         y, deterministic=deterministic)

    # return x + y

class Blocks(nn.Cell):

    def __init__(self):
        super(Blocks, self).__init__()
        self.block_23 = Block()
        self.block_16 = Block()
        self.block_6 = Block()
        self.block_11 = Block()
        self.block_5 = Block()
        self.block_12 = Block()
        self.block_2 = Block()
        self.block_0 = Block()
        self.block_9 = Block()
        self.block_3 = Block()
        self.block_7 = Block()
        self.block_1 = Block()
        self.block_19 = Block()
        self.block_13 = Block()
        self.block_18 = Block()
        self.block_10 = Block()
        self.block_17 = Block()
        self.block_20 = Block()
        self.block_8 = Block()
        self.block_21 = Block()
        self.block_14 = Block()
        self.block_4 = Block()
        self.block_15 = Block()
        self.block_22 = Block()

    def construct(self, x):
        block_23_opt = self.block_23(x)
        block_16_opt = self.block_16(block_23_opt)
        block_6_opt = self.block_6(block_16_opt)
        block_11_opt = self.block_11(block_6_opt)
        block_5_opt = self.block_5(block_11_opt)
        block_12_opt = self.block_12(block_5_opt)
        block_2_opt = self.block_2(block_12_opt)
        block_0_opt = self.block_0(block_2_opt)
        block_9_opt = self.block_9(block_0_opt)
        block_3_opt = self.block_3(block_9_opt)
        block_7_opt = self.block_7(block_3_opt)
        block_1_opt = self.block_1(block_7_opt)
        block_19_opt = self.block_19(block_1_opt)
        block_13_opt = self.block_13(block_19_opt)
        block_18_opt = self.block_18(block_13_opt)
        block_10_opt = self.block_10(block_18_opt)
        block_17_opt = self.block_17(block_10_opt)
        block_20_opt = self.block_20(block_17_opt)
        block_8_opt = self.block_8(block_20_opt)
        block_21_opt = self.block_21(block_8_opt)
        block_14_opt = self.block_14(block_21_opt)
        block_4_opt = self.block_4(block_14_opt)
        block_15_opt = self.block_15(block_4_opt)
        block_22_opt = self.block_22(block_15_opt)
        return block_22_opt


class MindSporeModel(nn.Cell):

    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.patchembed_0 = PatchEmbed()
        self.broadcast_to_4_shape1 = 1
        self.broadcast_to_4_shape2 = 1024
        self.broadcast_to_4_input = Parameter(Tensor(np.random.uniform(0, 1, (1, 1, 1024)).astype(np.float32)),
                                              name=None)
        self.concat_5 = P.Concat(axis=1)
        self.add_6_alpha = 1
        self.add_6_input_1 = Parameter(Tensor(np.random.uniform(0, 1, (1, 577, 1024)).astype(np.float32)), name=None)
        self.dropout_7 = nn.Dropout()
        self.blocks_0 = Blocks()
        self.layernorm_704 = nn.LayerNorm(normalized_shape=(1024, ), epsilon=1e-06)
        self.prim_slice_705_starts = 0
        self.prim_slice_705_steps = 1
        self.linear_707 = nn.Dense()

    def construct(self, x_1):
        patchembed_0_opt = self.patchembed_0(x_1)
        opt_shape_3 = P.Shape()(patchembed_0_opt)[0]
        opt_broadcast_to_4 = ms_np.broadcast_to(self.broadcast_to_4_input,
                                                (opt_shape_3, self.broadcast_to_4_shape1, self.broadcast_to_4_shape2))
        opt_concat_5 = self.concat_5((opt_broadcast_to_4, patchembed_0_opt))
        opt_add_6 = opt_concat_5 + self.add_6_input_1
        opt_dropout_7 = self.dropout_7(opt_add_6)
        blocks_0_opt = self.blocks_0(opt_dropout_7)
        opt_layernorm_704 = self.layernorm_704(blocks_0_opt)
        opt_prim_slice_705 = opt_layernorm_704[self.prim_slice_705_starts::self.prim_slice_705_steps, :, :]
        opt_primop_706 = opt_prim_slice_705[:, 0, :]
        opt_linear_707 = self.linear_707(opt_primop_706)
        return opt_linear_707
