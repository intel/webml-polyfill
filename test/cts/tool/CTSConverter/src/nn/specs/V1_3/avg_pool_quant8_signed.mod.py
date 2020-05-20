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

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 1}, 0.5f, -128")
cons1 = Int32Scalar("cons1", 1)
pad0 = Int32Scalar("pad0", 0)
act = Int32Scalar("act", 0)
o = Output("op3", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 2, 1}, 0.5f, -128")
model = model.Operation("AVERAGE_POOL_2D", i1, pad0, pad0, pad0, pad0, cons1, cons1, cons1, cons1, act).To(o)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-127, -126, -125, -124]}

output0 = {o: # output 0
          [-127, -126, -125, -124]}

# Instantiate an example
Example((input0, output0))

#######################################################

model = Model()

bat = 5
row = 52
col = 60
chn = 3

i0 = Input("i0", "TENSOR_QUANT8_ASYMM_SIGNED", "{%d, %d, %d, %d}, 0.5f, -128" % (bat, row, col, chn))

std = 5
flt = 10
pad = 5

stride = Int32Scalar("stride", std)
filt = Int32Scalar("filter", flt)
padding = Int32Scalar("padding", pad)
act0 = Int32Scalar("activation", 0)
output_row = (row + 2 * pad - flt + std) // std
output_col = (col + 2 * pad - flt + std) // std

output = Output("output", "TENSOR_QUANT8_ASYMM_SIGNED",
                "{%d, %d, %d, %d}, 0.5f, -128" % (bat, output_row, output_col, chn))

model = model.Operation(
    "AVERAGE_POOL_2D", i0, padding, padding, padding, padding, stride, stride, filt, filt, act0).To(output)

# Example 1. Input in operand 0,
input_values = [127 for _ in range(bat * row * col * chn)]
input0 = {i0: input_values}
output_values = [127 for _ in range(bat * output_row * output_col * chn)]
output0 = {output: output_values}

# Instantiate an example
Example((input0, output0))

#######################################################

model = Model()

bat = 1
row = 100
col = 100
chn = 1

i0 = Input("i0", "TENSOR_QUANT8_ASYMM_SIGNED", "{%d, %d, %d, %d}, 0.5f, -128" % (bat, row, col, chn))

std = 4
flt = 10
pad = 0

stride = Int32Scalar("stride", std)
filt = Int32Scalar("filter", flt)
padding = Int32Scalar("padding", pad)
act0 = Int32Scalar("activation", 0)
output_row = (row + 2 * pad - flt + std) // std
output_col = (col + 2 * pad - flt + std) // std

output = Output("output", "TENSOR_QUANT8_ASYMM_SIGNED",
                "{%d, %d, %d, %d}, 0.5f, -128" % (bat, output_row, output_col, chn))

model = model.Operation(
    "AVERAGE_POOL_2D", i0, padding, padding, padding, padding, stride, stride, filt, filt, act0).To(output)

# Example 1. Input in operand 0,
input_values = [x % 4 * 2 - 128 for x in range(bat * row * col * chn)]
input0 = {i0: input_values}
output_values = [-125 for _ in range(bat * output_row * output_col * chn)]
output0 = {output: output_values}

# Instantiate an example
Example((input0, output0))

#######################################################

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 3, 1}, 0.5f, -128")
cons1 = Int32Scalar("cons1", 1)
pad0 = Int32Scalar("pad0", 0)
act2 = Int32Scalar("relu1_activitation", 2)
o = Output("op3", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 3, 3, 1}, 0.5f, -128")
model = model.Operation("AVERAGE_POOL_2D", i1, pad0, pad0, pad0, pad0, cons1, cons1, cons1, cons1, act2).To(o)

# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-128, -127, -126, -125, -124, -123, -122, -121, -120]}

output0 = {o: # output 0
          [-128, -127, -126, -126, -126, -126, -126, -126, -126]}

# Instantiate an example
Example((input0, output0))

#######################################################

model = Model()
i1 = Input("op1", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 2, 4, 1}, 0.0625f, -128") # input 0
cons2 = Int32Scalar("cons2", 2)
pad_same = Int32Scalar("pad_same", 1)
act_none = Int32Scalar("act_none", 0)
i3 = Output("op3", "TENSOR_QUANT8_ASYMM_SIGNED", "{1, 1, 2, 1}, 0.0625f, -128") # output 0
model = model.Operation("AVERAGE_POOL_2D", i1, pad_same, cons2, cons2, cons2, cons2, act_none).To(i3)
# Example 1. Input in operand 0,
input0 = {i1: # input 0
          [-128, -32, -96, -64, -80, -96, 32, -16]}
output0 = {i3: # output 0
          [-84, -36]}
# Instantiate an example
Example((input0, output0))

#######################################################

layout = BoolScalar("layout", False) # NHWC

# TEST 1: AVERAGE_POOL_2D_NCHW_1, pad = 0, stride = 1, filter = 1, act = none
i1 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
o1 = Output("op4", "TENSOR_FLOAT32", "{1, 2, 2, 1}")
Model().Operation("AVERAGE_POOL_2D", i1, 0, 0, 0, 0, 1, 1, 1, 1, 0, layout).To(o1)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128)
})

# Instantiate an example
example = Example({
    i1: [1.0, 2.0, 3.0, 4.0],
    o1: [1.0, 2.0, 3.0, 4.0]
}).AddNchw(i1, o1, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# TEST 2: AVERAGE_POOL_2D_NCHW_2, act = none
bat = 5
row = 52
col = 60
chn = 3
std = 5
flt = 100
pad = 50
output_row = (row + 2 * pad - flt + std) // std
output_col = (col + 2 * pad - flt + std) // std

i2 = Input("op1", ("TENSOR_FLOAT32", [bat, row, col, chn]))
o2 = Output("op4", ("TENSOR_FLOAT32", [bat, output_row, output_col, chn]))
Model().Operation("AVERAGE_POOL_2D", i2, pad, pad, pad, pad, std, std, flt, flt, 0, layout).To(o2)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    o2: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128)
})

# Instantiate an example
example = Example({
    i2: [1. for _ in range(bat * row * col * chn)],
    o2: [1. for _ in range(bat * output_row * output_col * chn)]
}).AddNchw(i2, o2, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# TEST 3: AVERAGE_POOL_2D_NCHW_3, act = none
bat = 1
row = 200
col = 180
chn = 1
std = 2
flt = 10
pad = 0
output_row = (row + 2 * pad - flt + std) // std
output_col = (col + 2 * pad - flt + std) // std

i3 = Input("op1", ("TENSOR_FLOAT32", [bat, row, col, chn]))
o3 = Output("op4", ("TENSOR_FLOAT32", [bat, output_row, output_col, chn]))
Model().Operation("AVERAGE_POOL_2D", i3, pad, pad, pad, pad, std, std, flt, flt, 0, layout).To(o3)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.25, -128),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.25, -128)
})

# Instantiate an example
example = Example({
    i3: [x % 2 for x in range(bat * row * col * chn)],
    o3: [.5 for _ in range(bat * output_row * output_col * chn)]
}).AddNchw(i3, o3, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# TEST 4: AVERAGE_POOL_2D_NCHW_4, act = relu6
bat = 5
row = 52
col = 60
chn = 3
std = 5
flt = 100
pad = 50
output_row = (row + 2 * pad - flt + std) // std
output_col = (col + 2 * pad - flt + std) // std

i4 = Input("op1", ("TENSOR_FLOAT32", [bat, row, col, chn]))
o4 = Output("op4", ("TENSOR_FLOAT32", [bat, output_row, output_col, chn]))
Model().Operation("AVERAGE_POOL_2D", i4, pad, pad, pad, pad, std, std, flt, flt, 3, layout).To(o4)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i4: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128),
    o4: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.5, -128)
})

# Instantiate an example
example = Example({
    i4: [10 for _ in range(bat * row * col * chn)],
    o4: [6 for _ in range(bat * output_row * output_col * chn)]
}).AddNchw(i4, o4, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# TEST 5: AVERAGE_POOL_2D_NCHW_5, pad = same, stride = 2, filter = 2, act = none
i5 = Input("op1", "TENSOR_FLOAT32", "{1, 2, 4, 1}")
o5 = Output("op4", "TENSOR_FLOAT32", "{1, 1, 2, 1}")
Model().Operation("AVERAGE_POOL_2D", i5, 1, 2, 2, 2, 2, 0, layout).To(o5)

# Additional data type
quant8_signed = DataTypeConverter().Identify({
    i5: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.25, -128),
    o5: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.25, -128)
})

# Instantiate an example
example = Example({
    i5: [0, 6, 2, 4, 3, 2, 10, 7],
    o5: [2.75, 5.75]
}).AddNchw(i5, o5, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# TEST 6: zero-sized input, explicit padding

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

# AVERAGE_POOL_2D op with numBatches = 0.
o3 = Output("out", "TENSOR_FLOAT32", "{0, 1, 1, 1}") # out
model = model.Operation("AVERAGE_POOL_2D", zero_sized, 0, 0, 0, 0, 1, 1, 2, 2, 0, layout).To(o3)

quant8_signed = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    zero_sized: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0)
})

Example({
    i1: [1],
    o1: [],
    o2: [],
    o3: [],
}).AddNchw(i1, zero_sized, o3, layout).AddVariations(quant8_signed, includeDefault=False)

#######################################################

# TEST 7: zero-sized input, implicit padding

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

# AVERAGE_POOL_2D op with numBatches = 0.
o3 = Output("out", "TENSOR_FLOAT32", "{0, 2, 2, 1}") # out
model = model.Operation("AVERAGE_POOL_2D", zero_sized, 1, 1, 1, 2, 2, 0, layout).To(o3)

quant8_signed = DataTypeConverter().Identify({
    p1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    p2: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    o1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    tmp1: ("TENSOR_QUANT16_ASYMM", 0.125, 0),
    i1: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    zero_sized: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0),
    o3: ("TENSOR_QUANT8_ASYMM_SIGNED", 0.1, 0)
})

Example({
    i1: [1],
    o1: [],
    o2: [],
    o3: [],
}).AddNchw(i1, zero_sized, o3, layout).AddVariations(quant8_signed, includeDefault=False)
