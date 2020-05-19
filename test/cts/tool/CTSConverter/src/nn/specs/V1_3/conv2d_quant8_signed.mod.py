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
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}", [.25, .25, .25, .25])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
Model().Operation("CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 0, layout, 1, 1).To(o1)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128),
    b1: ("TENSOR_INT32", 0.0625, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})

# Instantiate an example
example = Example({
    i1: [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
    o1: [.875, .875, .875, .875]
}).AddNchw(i1, o1, layout).AddVariations(quant8_signed, includeDefault=False)


# dilation set to 3
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 9, 9, 1}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [1, 2, 3, 4, 5, 6, 7, 8, 9])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
Model().Operation("CONV_2D", i2, f2, b2, 0, 0, 0, 0, 1, 1, 0, layout, 3, 3).To(o2)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128),
    b2: ("TENSOR_INT32", 0.0625, 0),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})

# Instantiate an example
example = Example({
    i2: [0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0],
    o2: [5, 5, 5, 5, 5, 5, 5, 5, 5]
}).AddNchw(i2, o2, layout).AddVariations(quant8_signed, includeDefault=False)

# same as test 1 but with implicit VALID padding
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}", [.25, .25, .25, .25])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
Model().Operation("CONV_2D", i1, f1, b1, 2, 1, 1, 0, layout, 1, 1).To(o1)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128),
    b1: ("TENSOR_INT32", 0.0625, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})

# Instantiate an example
example = Example({
    i1: [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
    o1: [.875, .875, .875, .875]
}, name="valid_padding").AddNchw(i1, o1, layout).AddVariations(quant8_signed, includeDefault=False)


# same as test 2 but with implicit VALID padding
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 9, 9, 1}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [1, 2, 3, 4, 5, 6, 7, 8, 9])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
Model().Operation("CONV_2D", i2, f2, b2, 2, 1, 1, 0, layout, 3, 3).To(o2)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128),
    b2: ("TENSOR_INT32", 0.0625, 0),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})

# Instantiate an example
example = Example({
    i2: [0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0],
    o2: [5, 5, 5, 5, 5, 5, 5, 5, 5]
}, name="valid_padding").AddNchw(i2, o2, layout).AddVariations(quant8_signed, includeDefault=False)


# dilation set to 3, SAME padding
i3 = Input("op1", "TENSOR_FLOAT32", "{1, 6, 6, 1}")
f3 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}", [1, 2, 3, 4])
b3 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o3 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
Model().Operation("CONV_2D", i3, f3, b3, 1, 2, 2, 0, layout, 3, 3).To(o3)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128),
    b3: ("TENSOR_INT32", 0.0625, 0),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})

# Instantiate an example
example = Example({
    i3: [0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 4, 3, 0, 0,
         0, 0, 2, 1, 0, 0,
         0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0],
    o3: [16, 0, 9, 0, 0, 0, 4, 0, 1]
}).AddNchw(i3, o3, layout).AddVariations(quant8_signed, includeDefault=False)

# No layout param specified
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 1, 2}, 0.5f, 0")
f1 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{3, 1, 1, 2}",
               [1, 2, 1, 2, 1, 2], extraParams = SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 0.75, 1.0]))
b1 = Parameter("op3", "TENSOR_INT32", "{3}", [4, 4, 4])
o1 = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 1, 3}, 1.f, 0")
Model().Operation("CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 0).To(o1)

# Instantiate an example
Example({
    i1: [10, 10, 10, 10, 10, 10],
    o1: [9, 13, 17, 9, 13, 17, 9, 13, 17]
})

# layout param, NHWC/NCHW layouts
layout = BoolScalar("layout", False) # NHWC
i2 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 1, 2}, 0.5f, 0")
f2 = Parameter("op2", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{3, 1, 1, 2}",
               [1, 2, 1, 2, 1, 2], extraParams = SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 0.75, 1.0]))
b2 = Parameter("op3", "TENSOR_INT32", "{3}", [4, 4, 4])
o2 = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 1, 3}, 1.f, 0")
Model("layouts").Operation("CONV_2D", i2, f2, b2, 0, 0, 0, 0, 1, 1, 0, layout).To(o2)

# Instantiate an example
Example({
    i2: [10, -20, 10, -20, 10, -20],
    o2: [-7, -10, -13, -7, -10, -13, -7, -10, -13]
}).AddNchw(i2, o2, layout)

# zero-sized input

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2}, 0.1f, 0", [9, 1]) # scores
p2 = Parameter("roi", "TENSOR_QUANT16_ASYMM", "{1, 8}, 0.125f, 0", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_QUANT8_ASYMM_SIGNED", "{0}, 0.1f, 0") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_QUANT16_ASYMM", "{0, 4}, 0.125f, 0") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3, -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
i1 = Input("in", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 1, 2}, 0.5f, 0")
zero_sized = Internal("featureMap", "TENSOR_QUANT8_ASYMM_SIGNED", "{0, 2, 2, 2}, 0.5f, 0")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# CONV_2D op with numBatches = 0.
w = Parameter("weights", "TENSOR_QUANT8_SYMM_PER_CHANNEL", "{3, 1, 1, 2}",
              [1, 2, 1, 2, 1, 2], extraParams = SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 0.75, 1.0]))
b = Parameter("bias", "TENSOR_INT32", "{3}", [4, 4, 4])
o3 = Output("out", "TENSOR_QUANT8_ASYMM_SIGNED", "{0, 2, 2, 3}, 1.f, 0") # out
model = model.Operation("CONV_2D", zero_sized, w, b, 0, 0, 0, 0, 1, 1, 0, layout).To(o3)

Example({
    i1: [2, 2],
    o1: [],
    o2: [],
    o3: [],
}).AddNchw(i1, zero_sized, o3, layout)

layout = BoolScalar("layout", False) # NHWC

# CONV_NCHW_1
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 3, 1}")
f1 = Parameter("op2", "TENSOR_FLOAT32", "{1, 2, 2, 1}", [.25, .25, .25, .25])
b1 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [0])
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
Model().Operation("CONV_2D", i1, f1, b1, 0, 0, 0, 0, 1, 1, 0, layout).To(o1)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128),
    b1: ("TENSOR_INT32", 0.0625, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})
channelquant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f1: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.125])),
    b1: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.0625], hide=True)),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.125, -128)
})

# Instantiate an example
example = Example({
    i1: [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
    o1: [.875, .875, .875, .875]
}).AddNchw(i1, o1, layout).AddVariations(quant8_signed, channelquant8_signed, includeDefault=False)


# CONV_NCHW_2
i2 = Input("op1", "TENSOR_FLOAT32", "{1, 3, 4, 1}")
f2 = Parameter("op2", "TENSOR_FLOAT32", "{1, 3, 3, 1}", [1, 4, 7, 2, 5, 8, 3, 6, 9])
b2 = Parameter("op3", "TENSOR_FLOAT32", "{1}", [-200])
o2 = Output("op4", "TENSOR_FLOAT32", "{1, 3, 4, 1}")
Model().Operation("CONV_2D", i2, f2, b2, 1, 1, 1, 1, layout).To(o2)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -1),
    f2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -1),
    b2: ("TENSOR_INT32", 0.25, 0),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 1.0, -78)
})
channelquant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -1),
    f2: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5])),
    b2: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.25], hide=True)),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 1.0, -78)
})

# Instantiate an example
example = Example({
    i2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    o2: [0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0]
}).AddNchw(i2, o2, layout).AddVariations(quant8_signed, channelquant8_signed, includeDefault=False)


# CONV_NCHW_CHANNEL
i3 = Input("op1", "TENSOR_FLOAT32", "{1, 1, 1, 3}")
f3 = Parameter("op2", "TENSOR_FLOAT32", "{3, 1, 1, 3}", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
b3 = Parameter("op3", "TENSOR_FLOAT32", "{3}", [0., 0., 0.])
o3 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 1, 3}")
Model("channel").Operation("CONV_2D", i3, f3, b3, 0, 0, 0, 0, 1, 1, 0, layout).To(o3)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    b3: ("TENSOR_INT32", 0.25, 0),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128)
})
channelquant8_signed = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    f3: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 0.4, 0.3])),
    b3: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.25, 0.2, 0.15], hide=True)),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128)
})

# Instantiate an example
example = Example({
    i3: [5.,  5.,   5.],
    o3: [15., 37.5, 60.]
}).AddNchw(i3, o3, layout).AddVariations(quant8_signed, channelquant8_signed, includeDefault=False)


# CONV_NCHW_LARGE
i4 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 3, 3}")
f4 = Parameter("op2", "TENSOR_FLOAT32", "{3, 1, 1, 3}", [1., 4., 7., 2., 5., 8., 3., 6., 9.])
b4 = Parameter("op3", "TENSOR_FLOAT32", "{3}", [0., 0., 0.])
o4 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 3, 3}")
Model("large").Operation("CONV_2D", i4, f4, b4, 0, 0, 0, 0, 1, 1, 0, layout).To(o4)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    f4: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    b4: ("TENSOR_INT32", 0.25, 0),
    o4: ("TENSOR_QUANT8_ASYMM_SIGNED", 2.0, -128)
})
channelquant8_signed = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, 0),
    f4: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 1.0, 0.5])),
    b4: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.25, 0.5, 0.25], hide=True)),
    o4: ("TENSOR_QUANT8_ASYMM_SIGNED", 2.0, -128)
})
channelQuant8_mult_gt_1 = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM_SIGNED", 1.0, -1),
    f4: ("TENSOR_QUANT8_SYMM_PER_CHANNEL", 0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 1.0, 1.005])),
    b4: ("TENSOR_INT32", 0.0, 0, SymmPerChannelQuantParams(channelDim=0, scales=[0.5, 1.0, 1.005], hide=True)),
    o4: ("TENSOR_QUANT8_ASYMM_SIGNED", 1.0, -1)
})

# Instantiate an example
example = Example({
    i4: [1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
         10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.],
    o4: [30.,   36.,   42.,
         66.,   81.,   96.,
         102.,  126.,  150.,
         138.,  171.,  204.,
         174.,  216.,  258.,
         210.,  261.,  312.]
}).AddNchw(i4, o4, layout).AddVariations(quant8_signed, channelquant8_signed, channelQuant8_mult_gt_1, includeDefault=False)

# quantized with scale product greater than output scale
scale = 256.5 / 255
zero_point = 0
i9 = Input("op1", ("TENSOR_QUANT8_ASYMM_SIGNED", [2, 2, 4, 1], scale, zero_point))
f9 = Parameter("op2", ("TENSOR_QUANT8_ASYMM_SIGNED", [3, 2, 2, 1], scale, zero_point),
               [1, 2, 3, 4, -1, 1, -1, 1, -1, -1, 1, 1])
b9 = Parameter("op3", ("TENSOR_INT32", [3], scale * scale, 0), [1, 2, 3])
o9 = Output("op4", ("TENSOR_QUANT8_ASYMM_SIGNED", [2, 1, 2, 3], 1.0, -1))
model9 = Model("quant_output_multiplier_gt_1").Operation("CONV_2D", i9, f9, b9, 2, 2, 2, 0).To(o9)

# Instantiate an example
example = Example({
    i9: [
        1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 3, 4, 1, 2,
        3, 4
    ],
    o9: [17, 1, 4, 17, 1, 4, 16, 3, 2, 36, 3, 2]
}, model=model9)


# zero-sized input, explicit padding

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_FLOAT32", "{1, 2}", [0.90, 0.10]) # scores
p2 = Parameter("roi", "TENSOR_FLOAT32", "{1, 8}", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_FLOAT32", "{0}") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_FLOAT32", "{0, 4}") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3, -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 1}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# CONV_2D op with numBatches = 0.
w = Parameter("weights", "TENSOR_FLOAT32", "{2, 1, 1, 1}", [3, 4]) # weights
b = Parameter("bias", "TENSOR_FLOAT32", "{2}", [1, 2]) # bias
o3 = Output("out", "TENSOR_FLOAT32", "{0, 2, 2, 2}") # out
model = model.Operation("CONV_2D", zero_sized, w, b, 0, 0, 0, 0, 1, 1, 0, layout).To(o3)

quant8_signed = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    zero_sized: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    w: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    b: ("TENSOR_INT32", 0.01, 0),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0)
})

Example({
    i1: [1],
    o1: [],
    o2: [],
    o3: [],
}).AddNchw(i1, zero_sized, o3, layout).AddVariations(quant8_signed, includeDefault=False)


# zero-sized input, implicit padding

# Use BOX_WITH_NMS_LIMIT op to generate a zero-sized internal tensor for box cooridnates.
p1 = Parameter("scores", "TENSOR_FLOAT32", "{1, 2}", [0.90, 0.10]) # scores
p2 = Parameter("roi", "TENSOR_FLOAT32", "{1, 8}", [1, 1, 10, 10, 0, 0, 10, 10]) # roi
o1 = Output("scoresOut", "TENSOR_FLOAT32", "{0}") # scores out
o2 = Output("classesOut", "TENSOR_INT32", "{0}") # classes out
tmp1 = Internal("roiOut", "TENSOR_FLOAT32", "{0, 4}") # roi out
tmp2 = Internal("batchSplitOut", "TENSOR_INT32", "{0}") # batch split out
model = Model("zero_sized").Operation("BOX_WITH_NMS_LIMIT", p1, p2, [0], 0.3, -1, 0, 0.4, 1.0, 0.3).To(o1, tmp1, o2, tmp2)

# Use ROI_ALIGN op to convert into zero-sized feature map.
i1 = Input("in", "TENSOR_FLOAT32", "{1, 1, 1, 1}")
zero_sized = Internal("featureMap", "TENSOR_FLOAT32", "{0, 2, 2, 1}")
model = model.Operation("ROI_ALIGN", i1, tmp1, tmp2, 2, 2, 2.0, 2.0, 4, 4, layout).To(zero_sized)

# CONV_2D op with numBatches = 0.
w = Parameter("weights", "TENSOR_FLOAT32", "{2, 1, 1, 1}", [3, 4]) # weights
b = Parameter("bias", "TENSOR_FLOAT32", "{2}", [1, 2]) # bias
o3 = Output("out", "TENSOR_FLOAT32", "{0, 2, 2, 2}") # out
model = model.Operation("CONV_2D", zero_sized, w, b, 1, 1, 1, 0, layout).To(o3)

quant8_signed = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    zero_sized: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    w: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    b: ("TENSOR_INT32", 0.01, 0),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0)
})

Example({
    i1: [1],
    o1: [],
    o2: [],
    o3: [],
}).AddNchw(i1, zero_sized, o3, layout).AddVariations(quant8_signed, includeDefault=False)

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 6, 1}, 0.5f, -1")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 1}, 0.5f, -1",
               [1, 3, 5, 7])
b1 = Parameter("op3", "TENSOR_INT32", "{1}, 0.25f, 0", [-4])
pad_valid = Int32Scalar("pad_valid", 2)
act_none = Int32Scalar("act_none", 0)
stride1 = Int32Scalar("stride1", 1)
stride3 = Int32Scalar("stride3", 3)

output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 1}, 1.f, -1")

model = model.Operation("CONV_2D", i1, f1, b1, pad_valid, stride3,
                        stride1, act_none).To(output)

# Example 1. Input in operand 0,
input0 = {
    i1:  # input 0
        [5, 3, 1, -3, -5, -7,
         7, 5, 3, -5, -7, -9,
         9, 7, 5, -7, -9, -11]
}

output0 = {
    output:  # output 0
        [29, -25, 39, -35]
}

# Instantiate an example
Example((input0, output0))

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 1, 3}, 0.5f, -128")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{3, 1, 1, 3}, 0.5f, -128", [-127, -126, -125, -124, -123, -122, -121, -120, -119])
b1 = Parameter("op3", "TENSOR_INT32", "{3}, 0.25, 0", [0, 0, 0])
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 1, 3}, 1.0, -128")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-118, -118, -118]}

output0 = {output: # output 0
           [-113, -90, -68]}

# Instantiate an example
Example((input0, output0))

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 1, 3}, 0.5f, -128")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{3, 1, 1, 3}, 0.5f, -128")
b1 = Input("op3", "TENSOR_INT32", "{3}, 0.25, 0")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 1, 3}, 1.0, -128")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-118, -118, -118],
          f1:
          [-127, -126, -125,
           -124, -123, -122,
           -121, -120, -119],
          b1:
          [0, 0, 0]}

output0 = {output: # output 0
           [-113, -90, -68]}

# Instantiate an example
Example((input0, output0))

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 3, 3}, 0.5, -128")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{3, 1, 1, 3}, 0.5, -128", [-127, -124, -121, -126, -123, -120, -125, -122, -119])
b1 = Parameter("op3", "TENSOR_INT32", "{3}, 0.25, 0", [0, 0, 0])
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
output = Output("op4",  "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 3, 3}, 1.0, -128")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [  -127,   -126,   -125,   -124,   -123,   -122,   -121,   -120,   -119,
            -118,  -117,  -116,  -115,  -114,  -113,  -112,  -111,  -110]}

output0 = {output: # output 0
           [  -120,   -119,   -117,
              -111,  -107,  -104,
              -102,  -96,  -90,
              -93,  -85,  -77,
              -84,  -74,  -63,
              -75,  -62,  -50]
          }

# Instantiate an example
Example((input0, output0))

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 3, 3}, 0.5, -128")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{3, 1, 1, 3}, 0.5, -128")
b1 = Input("op3", "TENSOR_INT32", "{3}, 0.25, 0")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
output = Output("op4",  "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 3, 3}, 1.0, -128")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [  -127,   -126,   -125,   -124,   -123,   -122,   -121,   -120,   -119,
             -118,  -117,  -116,  -115,  -114,  -113,  -112,  -111,  -110],
          f1:
          [ -127,  -124,  -121,
            -126,  -123,  -120,
            -125,  -122,  -119],
          b1:
          [0, 0, 0]}

output0 = {output: # output 0
           [  -120,   -119,   -117,
              -111,  -107,  -104,
              -102,  -96,  -90,
              -93,  -85,  -77,
              -84,  -74,  -63,
              -75,  -62,  -50]
          }

# Instantiate an example
Example((input0, output0))

# conv_quant8.mod.py with biases and filter being constants

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 3, 1}, 0.5f, -128")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 1}, 0.5f, -128",
               [-126, -126, -126, -126])
b1 = Parameter("op3", "TENSOR_INT32", "{1}, 0.25f, 0", [4])
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
# output dimension:
#     (i1.height - f1.height + 1) x (i1.width - f1.width + 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 1}, 1.f, -128")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride,
                        stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {
    i1:  # input 0
        [-120, -120, -120, -120, -124, -120, -120, -120, -120]
}
# (i1 (conv) f1) + b1
output0 = {
    output:  # output 0
        [-113, -113, -113, -113]
}

# Instantiate an example
Example((input0, output0))

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 3, 3}, 0.5, -128")
f1 = Parameter("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{3, 1, 1, 3}, 0.5, -128",
               [-118, -88, -58, -108, -78, -48, -98, -68, -38])
b1 = Parameter("op3", "TENSOR_INT32", "{3}, 0.25, 0", [0, 0, 0])
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
output = Output("op4",  "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 3, 3}, 1.0, -128")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [  -127,   -126,   -125,   -124,   -123,   -122,   -121,   -120,   -119,
            -118,  -117,  -116,  -115,  -114,  -113,  -112,  -111,  -110]}

output0 = {output: # output 0
           [  -53,  -38,  -23,
              37, 75, 112,
              127, 127, 127,
              127, 127, 127,
              127, 127, 127,
              127, 127, 127]
          }

# Instantiate an example
Example((input0, output0))

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 3, 1}, 0.5f, -128")
f1 = Input("op2", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 1}, 0.5f, -128")
b1 = Input("op3", "TENSOR_INT32", "{1}, 0.25f, 0")
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
stride = Int32Scalar("stride", 1)
# output dimension:
#     (i1.height - f1.height + 1) x (i1.width - f1.width + 1)
output = Output("op4", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 1}, 1.f, -128")

model = model.Operation("CONV_2D", i1, f1, b1, pad0, pad0, pad0, pad0, stride, stride, act).To(output)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-120, -120, -120, -120, -124, -120, -120, -120, -120],
          f1:
          [-126, -126, -126, -126],
          b1:
          [4]}
# (i1 (conv) f1) + b1
output0 = {output: # output 0
           [-113, -113, -113, -113]}

# Instantiate an example
Example((input0, output0))
