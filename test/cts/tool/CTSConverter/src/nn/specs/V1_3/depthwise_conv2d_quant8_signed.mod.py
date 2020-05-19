#
# Copyright (C) 2019 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

layout = BoolScalar("layout", False) # NHWC

# dilation set to 1 (default)
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 2}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [.25, 0., .2, 0., .25, 0., 0., .3, .25, 0., 0., 0., .25, .1, 0., 0.])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [1, 2, 3, 4])
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
Model().Operation("DEPTHWISE_CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 2, 0, layout, 1, 1).To(o1)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.01, -128),
    b1: ("TENSOR_INT32", 0.005, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, -128)
})

# Instantiate an example
example = Example({
    i1: [10, 21, 10, 22, 10, 23,
         10, 24, 10, 25, 10, 26,
         10, 27, 10, 28, 10, 29],
    o1: [11, 3, 7.2, 10.6,
         11, 3, 7.4, 10.9,
         11, 3, 7.8, 11.5,
         11, 3, 8.0, 11.8]
}).AddNchw(i1, o1, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# dilation set to 2
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 4, 2}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [0,0,0,0])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
Model().Operation("DEPTHWISE_CONV_2D", i2, f2, b2, 0, 0, 0, 0, 1, 1, 2, 0, layout, 2, 2).To(o2)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128),
    b2: ("TENSOR_INT32", 0.0625, 0),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})

# Instantiate an example
example = Example({
    i2: [0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,],
    o2: [13, 14, 0, 0,
         0, 0, 11, 12,
         5, 6, 0, 0,
         0, 0, 3, 4]
}).AddNchw(i2, o2, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# same as test 1 but with implicit padding
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 2}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [.25, 0., .2, 0., .25, 0., 0., .3, .25, 0., 0., 0., .25, .1, 0., 0.])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [1, 2, 3, 4])
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
Model().Operation("DEPTHWISE_CONV_2D", i1, f1, b1, 2, 1, 1, 2, 0, layout, 1, 1).To(o1)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.01, -128),
    b1: ("TENSOR_INT32", 0.005, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, -128)
})

# Instantiate an example
example = Example({
    i1: [10, 21, 10, 22, 10, 23,
         10, 24, 10, 25, 10, 26,
         10, 27, 10, 28, 10, 29],
    o1: [11, 3, 7.2, 10.6,
         11, 3, 7.4, 10.9,
         11, 3, 7.8, 11.5,
         11, 3, 8.0, 11.8]
}, name="valid_padding").AddNchw(i1, o1, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# same as test 2 but with implicit padding
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 4, 4, 2}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [0,0,0,0])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
Model().Operation("DEPTHWISE_CONV_2D", i2, f2, b2, 2, 1, 1, 2, 0, layout, 2, 2).To(o2)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, -128),
    b2: ("TENSOR_INT32", 0.05, 0),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, -128)
})

# Instantiate an example
example = Example({
    i2: [0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0,],
    o2: [13, 14, 0, 0,
         0, 0, 11, 12,
         5, 6, 0, 0,
         0, 0, 3, 4]
}, name="valid_padding").AddNchw(i2, o2, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# dilation set to 3, padding SAME, stride 2
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 6, 6, 1}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}", [1, 2, 3, 4])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
Model().Operation("DEPTHWISE_CONV_2D", i2, f2, b2, 1, 2, 2, 1, 0, layout, 3, 3).To(o2)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128),
    b2: ("TENSOR_INT32", 0.0625, 0),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})

# Instantiate an example
example = Example({
    i2: [0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 1, 1, 0, 0,
         0, 0, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0],
    o2: [4, 0, 3,
         0, 0, 0,
         2, 0, 1]
}, name="same_padding_stride_2").AddNchw(i2, o2, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# Same scales, zeroPoint = 0
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128")
f1 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{1, 2, 2, 2}",
               [2, 4,  2, 0,  2, 2,  2, 0],
               extraParams = SymmPerChannelQuantParams(channelDim=3, scales=[0.5, 0.5]))
b1 = Parameter("op3", "TENSOR_INT32", "{2}", [0, 0])
o1 = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 1, 2}, 1.f, -128")
Model("same").Operation("DEPTHWISE_CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 1, 0).To(o1)

# Instantiate an example
Example({
    i1: [-124, -112, -124, -96, -124, -64, -124, 0],
    o1: [-120, -80],
})

#######################################################

# Different scales, zeroPoint=128
i2 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 3, 2}, 0.5f, 0")
f2 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{1, 2, 2, 4}",
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               extraParams = SymmPerChannelQuantParams(channelDim=3, scales=[1.0, 0.5, 1.0, 0.5]))
b2 = Parameter("op3", "TENSOR_INT32", "{4}", [4, 4, 4, 4])
o2 = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 4}, 1.f, 0")
Model("different").Operation("DEPTHWISE_CONV_2D", i2, f2, b2, 0, 0, 0, 0, 1, 1, 2, 0).To(o2)

# Instantiate an example
Example({
    i2: [1, 2] * 9,
    o2: [4, 2, 6, 3, 4, 2, 6, 3,
         4, 2, 6, 3, 4, 2, 6, 3],
})

#######################################################

layout = BoolScalar("layout", False) # NHWC

# With layout param
i3 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 3, 2}, 0.5f, 0")
f3 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{1, 2, 2, 4}",
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               extraParams = SymmPerChannelQuantParams(channelDim=3, scales=[1.0, 0.5, 1.0, 0.5]))
b3 = Parameter("op3", "TENSOR_INT32", "{4}", [4, 4, 4, 4])
o3 = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 4}, 1.f, 0")
Model("layout").Operation("DEPTHWISE_CONV_2D", i3, f3, b3, 0, 0, 0, 0, 1, 1, 2, 0, layout).To(o3)

# Instantiate an example
Example({
    i3: [1, 2] * 9,
    o3: [4, 2, 6, 3, 4, 2, 6, 3,
         4, 2, 6, 3, 4, 2, 6, 3],
}).AddNchw(i3, o3, layout)

#######################################################

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 2, 2}, 0.5f, -1")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 4}, 0.5f, -1", [1, 3, 5, 7, -19, 19, -23, 23, 9, 11, 13, 15, 25, -29, 29, -33])
b1 = Parameter("op3", "TENSOR_INT32", "{4}, 0.25f, 0", [4, 8, 12, 16])
pad_valid = Int32Scalar("pad_valid", 2)
act_none = Int32Scalar("act_none", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 2)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 1, 4}, 1.f, -1")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad_valid,
                        stride, stride,
                        cm, act_none).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [1, 3, 13, 15,
           5, 7, 17, 19,
           9, 11, 21, 23]}
# (i1 (depthconv) f1)
output0 = {output: # output 0
           [70, -35, 98, -21,
            90, -27, 126, -5]}

# Instantiate an example
Example((input0, output0))

#######################################################

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128",
               [-126, -124, -126, -128, -126, -126, -126, -128])
b1 = Parameter("op3", "TENSOR_INT32", "{2}, 0.25f, 0", [0, 0])
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 1, 2}, 1.f, -128")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad0, pad0, pad0, pad0,
                        stride, stride,
                        cm, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-124, -112, -124, -96, -124, -64, -124, 0]}
# (i1 (depthconv) f1)
output0 = {output: # output 0
           [-120, -80]}

# Instantiate an example
Example((input0, output0))

#######################################################

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128")
b1 = Input("op3", "TENSOR_INT32", "{2}, 0.25f, 0")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 1, 2}, 1.f, -128")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad0, pad0, pad0, pad0,
                        stride, stride,
                        cm, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-124, -112, -124, -96, -124, -64, -124, 0],
          f1:
          [-126, -124,  -126, -128,  -126, -126,  -126, -128],
          b1:
          [0, 0]}
# (i1 (depthconv) f1)
output0 = {output: # output 0
           [-120, -80]}

# Instantiate an example
Example((input0, output0))

#######################################################

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128",
               [-126, -124, -126, -128, -126, -126, -126, -128])
b1 = Parameter("op3", "TENSOR_INT32", "{2}, 0.25f, 0", [0, 0])
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1,1,1,2}, 1.f, -128")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad0, pad0, pad0, pad0,
                        stride, stride,
                        cm, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-124, -112, -124, -96, -124, -64, -124, 0]}
# (i1 (depthconv) f1)
output0 = {output: # output 0
           [-120, -80]}

# Instantiate an example
Example((input0, output0))

#######################################################

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 2}, 0.5f, -128")
b1 = Input("op3", "TENSOR_INT32", "{2}, 0.25f, 0")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
cm = Int32Scalar("channelMultiplier", 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1,1,1,2}, 1.f, -128")

model = model.Operation("DEPTHWISE_CONV_2D",
                        i1, f1, b1,
                        pad0, pad0, pad0, pad0,
                        stride, stride,
                        cm, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-124, -112, -124, -96, -124, -64, -124, 0],
          f1:
          [-126, -124,  -126, -128,  -126, -126,  -126, -128],
          b1:
          [0, 0]}
# (i1 (depthconv) f1)
output0 = {output: # output 0
           [-120, -80]}

# Instantiate an example
Example((input0, output0))

#######################################################

layout = BoolScalar("layout", False) # NHWC

# DEPTHWISE_CONV2D_NCHW, pad = 0, stride = 1, cm = 2, act = none
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 2}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [.25, 0., .2, 0., .25, 0., 0., .3, .25, 0., 0., 0., .25, .1, 0., 0.])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [1, 2, 3, 4])
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
Model().Operation("DEPTHWISE_CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 2, 0, layout).To(o1)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.01, -128),
    b1: ("TENSOR_INT32", 0.005, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, -128)
})
channelquant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[0.01, 0.005, 0.01, 0.005])),
    b1: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.005, 0.0025, 0.005, 0.0025], hide=True)),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, -128)
})
channelQuant8_mult_gt_1 = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[0.01, 0.005, 0.01, 0.005])),
    b1: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.005, 0.0025, 0.005, 0.0025], hide=True)),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.0001, -128)
})

# Instantiate an example
example = Example({
    i1: [10, 21, 10, 22, 10, 23,
         10, 24, 10, 25, 10, 26,
         10, 27, 10, 28, 10, 29],
    o1: [11, 3, 7.2, 10.6,
         11, 3, 7.4, 10.9,
         11, 3, 7.8, 11.5,
         11, 3, 8.0, 11.8]
}).AddNchw(i1, o1, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# DEPTHWISE_CONV2D_NCHW_2, pad = valid, stride = 1, cm = 2, act = none
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 2, 2}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [1, 2, 3, 4])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 1, 4}")
Model().Operation("DEPTHWISE_CONV_2D", i2, f2, b2, 2, 1, 1, 2, 0, layout).To(o2)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    f2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    b2: ("TENSOR_INT32", 0.25, 0),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 1.0, -28)
})
channelquant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    f2: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[0.5, 0.25, 0.5, 0.25])),
    b2: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.25, 0.125, 0.25, 0.125], hide=True)),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 1.0, -28)
})

# Instantiate an example
example = Example({
    i2: [1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12],
    o2: [71, -34, 99, -20, 91, -26, 127, -4]
}).AddNchw(i2, o2, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# DEPTHWISE_CONV2D_NCHW_LARGE, pad = 0, stride = 1, cm = 1, act = none
i3 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 2}")
f3 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 2}", [.25, 0, .25, 1, .25, 0, .25, 1])
b3 = Parameter("op3", "TENSOR_FLOAT32", "{2}", [100, 200])
o3 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 1, 2}")
Model("large").Operation("DEPTHWISE_CONV_2D", i3, f3, b3, 0, 0, 0, 0, 1, 1, 1, 0, layout).To(o3)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -28),
    f3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, 0),
    b3: ("TENSOR_INT32", 0.0625, 0),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 2.0, 0)
})
channelquant8_signed = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    f3: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[0.125, 0.25])),
    b3: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.0625, 0.125], hide=True)),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 2.0, 0)
})

# Instantiate an example
example = Example({
    i3: [10, 21, 10, 22, 10, 23, 10, 24],
    o3: [110, 246]
}).AddNchw(i3, o3, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# DEPTHWISE_CONV2D_NCHW_LARGE, pad = 0, stride = 1, cm = 1, act = none
i4 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 4}")
f4 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 4}", [.25, 0, 10, 50, .25, 1, 20, 50, .25, 0, 30, 50, .25, 1, 40, 50])
b4 = Parameter("op3", "TENSOR_FLOAT32", "{4}", [6000, 7000, 8000, 9000])
o4 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 1, 4}")
Model("large").Operation("DEPTHWISE_CONV_2D", i4, f4, b4, 0, 0, 0, 0, 1, 1, 1, 0, layout).To(o4)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    f4: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.25, -128),
    b4: ("TENSOR_INT32", 0.125, 0),
    o4: ("TENSOR_QUANT8_ASYMM_SIGNED", 50.0, -128)
})
channelquant8_signed = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    f4: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=3, scales=[1.0, 2.0, 1.0, 1.0])),
    b4: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 1.0, 0.5, 0.5], hide=True)),
    o4: ("TENSOR_QUANT8_ASYMM_SIGNED", 50.0, -128)
})

# Instantiate an example
example = Example({
    i4: [10, 21, 10, 0,
         10, 22, 20, 0,
         10, 23, 30, 0,
         10, 24, 40, 0],
    o4: [6010, 7046, 11000, 9000]
}).AddNchw(i4, o4, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# quantized with scale product greater than output scale
input_scale = 256.5 / 255
input_zero_point = -1
filter_scale = 256.5 / 255
filter_zero_point = 0
i9 = Input("op1",
           ("TENSOR_QUANT8_ASYMM_SIGNED", [1, 3, 2, 2], input_scale, input_zero_point))
f9 = Parameter(
    "op2",
    ("TENSOR_QUANT8_ASYMM_SIGNED", [1, 2, 2, 4], filter_scale, filter_zero_point), [
        1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14,
        15, -16
    ])
b9 = Parameter("op3", ("TENSOR_INT32", [4], input_scale * filter_scale, 0),
               [2, 4, 6, 8])
o9 = Output("op4", ("TENSOR_QUANT8_ASYMM_SIGNED", [1, 2, 1, 4], 1.0, -1))
model9 = Model("quant_output_multiplier_gt_1").Operation("DEPTHWISE_CONV_2D", i9, f9, b9, 2, 1, 1, 2,
                           0).To(o9)

# Instantiate an example
example = Example({
    i9: [1, 3, 13, 15, 5, 7, 17, 19, 9, 11, 21, 23],
    o9: [127, -70, 127, -41, 127, -54, 127, -9]
}, model=model9)
