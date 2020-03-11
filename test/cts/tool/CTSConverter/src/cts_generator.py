#!/usr/bin/python3

# Copyright 2018, The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CTS testcase generator

Implements CTS test backend. Invoked by ml/nn/runtime/test/specs/generate_tests.sh;
See that script for details on how this script is used.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import math
import os
import re
import sys
import traceback

# Stuff from test generator
import test_generator as tg
from test_generator import ActivationConverter
from test_generator import BoolScalar
from test_generator import Configuration
from test_generator import DataTypeConverter
from test_generator import DataLayoutConverter
from test_generator import Example
from test_generator import Float16Scalar
from test_generator import Float32Scalar
from test_generator import Float32Vector
from test_generator import GetJointStr
from test_generator import IgnoredOutput
from test_generator import Input
from test_generator import Int32Scalar
from test_generator import Int32Vector
from test_generator import Internal
from test_generator import Model
from test_generator import Operand
from test_generator import Output
from test_generator import Parameter
from test_generator import ParameterAsInputConverter
from test_generator import RelaxedModeConverter
from test_generator import SmartOpen
from test_generator import SymmPerChannelQuantParams

def IndentedPrint(s, indent=2, *args, **kwargs):
    print('\n'.join([" " * indent + i for i in s.split('\n')]), *args, **kwargs)

# Take a model from command line
def ParseCmdLine():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="the spec file/directory")
    parser.add_argument(
        "-m", "--model", help="the output model file/directory", default="-")
    parser.add_argument(
        "-e", "--example", help="the output example file/directory", default="-")
    parser.add_argument(
        "-t", "--test", help="the output test file/directory", default="-")
    parser.add_argument(
        "-c", "--cts", help="the CTS TestGeneratedOneFile.cpp", default="-")
    parser.add_argument(
        "-f", "--force", help="force to regenerate all spec files", action="store_true")
    # for slicing tool
    parser.add_argument(
        "-l", "--log", help="the optional log file", default="")

    # For js
    parser.add_argument(
        "-js", "--jsTest", help="the output javascript file", default="-")
    # end

    args = parser.parse_args()

    ''' Original
    tg.FileNames.InitializeFileLists(
        args.spec, args.model, args.example, args.test, args.cts, args.log)
    '''

    # For js
    tg.FileNames.InitializeFileLists(
        args.spec, args.model, args.example, args.test, args.jsTest, args.cts, args.log)
    # end

    Configuration.force_regenerate = args.force

def NeedRegenerate():
    if not all(os.path.exists(f) for f in \
        [tg.FileNames.modelFile, tg.FileNames.exampleFile, tg.FileNames.testFile]):
        return True
    specTime = os.path.getmtime(tg.FileNames.specFile) + 10
    modelTime = os.path.getmtime(tg.FileNames.modelFile)
    exampleTime = os.path.getmtime(tg.FileNames.exampleFile)
    testTime = os.path.getmtime(tg.FileNames.testFile)
    if all(t > specTime for t in [modelTime, exampleTime, testTime]):
        return False
    return True

# For js
def NeedRegenerateForJS():
    if not os.path.exists(tg.FileNames.jsFile):
        return True
    specTime = os.path.getmtime(tg.FileNames.specFile) + 10
    jsTime = os.path.getmtime(tg.FileNames.jsFile)
    if jsTime > specTime:
        return False
    return True
# end

# Write headers for generated files, which are boilerplate codes only related to filenames
def InitializeFiles(model_fd, example_fd, test_fd):
    fileHeader = "// clang-format off\n// Generated file (from: {spec_file}). Do not edit"
    testFileHeader = """\
#include "../../TestGenerated.h"\n
namespace {spec_name} {{
// Generated {spec_name} test
#include "{example_file}"
// Generated model constructor
#include "{model_file}"
}} // namespace {spec_name}\n"""
    # This regex is to remove prefix and get relative path for #include
    pathRegex = r".*((frameworks/ml/nn/(runtime/test/)?)|(vendor/google/[a-z]*/test/))"
    specFileBase = os.path.basename(tg.FileNames.specFile)
    print(fileHeader.format(spec_file=specFileBase), file=model_fd)
    print(fileHeader.format(spec_file=specFileBase), file=example_fd)
    print(fileHeader.format(spec_file=specFileBase), file=test_fd)
    print(testFileHeader.format(
        model_file=re.sub(pathRegex, "", tg.FileNames.modelFile),
        example_file=re.sub(pathRegex, "", tg.FileNames.exampleFile),
        spec_name=tg.FileNames.specName), file=test_fd)

# For js
def InitializeFilesForJS(js_fd):
    fileHeader = "// Generated file (from: {spec_file}). Do not edit"
    specFileBase = os.path.basename(tg.FileNames.specFile)
    print(fileHeader.format(spec_file=specFileBase), file = js_fd)
    print ("describe('CTS', function() {", file = js_fd)
    print ("  const assert = chai.assert;", file = js_fd)
    print ("  const nn = navigator.ml.getNeuralNetworkContext();", file = js_fd)
# end

# Dump is_ignored function for IgnoredOutput
def DumpCtsIsIgnored(model, model_fd):
    isIgnoredTemplate = """\
inline bool {is_ignored_name}(int i) {{
  static std::set<int> ignore = {{{ignored_index}}};
  return ignore.find(i) != ignore.end();\n}}\n"""
    print(isIgnoredTemplate.format(
        ignored_index=tg.GetJointStr(model.GetIgnoredOutputs(), method=lambda x: str(x.index)),
        is_ignored_name=str(model.isIgnoredFunctionName)), file=model_fd)

# Dump Model file for Cts tests
def DumpCtsModel(model, model_fd):
    assert model.compiled
    if model.dumped:
        return
    print("void %s(Model *model) {"%(model.createFunctionName), file=model_fd)

    # Phase 0: types
    for t in model.GetTypes():
        if t.scale == 0.0 and t.zeroPoint == 0 and t.extraParams is None:
            typeDef = "OperandType %s(Type::%s, %s);"%(t, t.type, t.GetDimensionsString())
        else:
            if t.extraParams is None or t.extraParams.hide:
                typeDef = "OperandType %s(Type::%s, %s, %s, %d);"%(
                    t, t.type, t.GetDimensionsString(), tg.PrettyPrintAsFloat(t.scale), t.zeroPoint)
            else:
                typeDef = "OperandType %s(Type::%s, %s, %s, %d, %s);"%(
                    t, t.type, t.GetDimensionsString(), tg.PrettyPrintAsFloat(t.scale), t.zeroPoint,
                    t.extraParams.GetConstructor())

        IndentedPrint(typeDef, file=model_fd)

    # Phase 1: add operands
    print("  // Phase 1, operands", file=model_fd)
    for op in model.operands:
        IndentedPrint("auto %s = model->addOperand(&%s);"%(op, op.type), file=model_fd)

    # Phase 2: operations
    print("  // Phase 2, operations", file=model_fd)
    for p in model.GetParameters():
        paramDef = "static %s %s[] = %s;\nmodel->setOperandValue(%s, %s, sizeof(%s) * %d);"%(
            p.type.GetCppTypeString(), p.initializer, p.GetListInitialization(), p,
            p.initializer, p.type.GetCppTypeString(), p.type.GetNumberOfElements())
        IndentedPrint(paramDef, file=model_fd)
    for op in model.operations:
        IndentedPrint("model->addOperation(ANEURALNETWORKS_%s, {%s}, {%s});"%(
            op.optype, tg.GetJointStr(op.ins), tg.GetJointStr(op.outs)), file=model_fd)

    # Phase 3: add inputs and outputs
    print ("  // Phase 3, inputs and outputs", file=model_fd)
    IndentedPrint("model->identifyInputsAndOutputs(\n  {%s},\n  {%s});"%(
        tg.GetJointStr(model.GetInputs()), tg.GetJointStr(model.GetOutputs())), file=model_fd)

    # Phase 4: set relaxed execution if needed
    if (model.isRelaxed):
        print ("  // Phase 4: set relaxed execution", file=model_fd)
        print ("  model->relaxComputationFloat32toFloat16(true);", file=model_fd)

    print ("  assert(model->isValid());", file=model_fd)
    print ("}\n", file=model_fd)
    DumpCtsIsIgnored(model, model_fd)
    model.dumped = True

def DumpMixedType(operands, feedDict):
    supportedTensors = [
        "DIMENSIONS",
        "TENSOR_FLOAT32",
        "TENSOR_INT32",
        "TENSOR_QUANT8_ASYMM",
        "TENSOR_OEM_BYTE",
        "TENSOR_QUANT16_SYMM",
        "TENSOR_FLOAT16",
        "TENSOR_BOOL8",
        "TENSOR_QUANT8_SYMM_PER_CHANNEL",
        "TENSOR_QUANT16_ASYMM",
        "TENSOR_QUANT8_SYMM",
    ]
    typedMap = {t: [] for t in supportedTensors}
    FeedAndGet = lambda op, d: op.Feed(d).GetListInitialization()
    # group the operands by type
    for operand in operands:
        try:
            typedMap[operand.type.type].append(FeedAndGet(operand, feedDict))
            typedMap["DIMENSIONS"].append("{%d, {%s}}"%(
                operand.index, GetJointStr(operand.dimensions)))
        except KeyError as e:
            traceback.print_exc()
            sys.exit("Cannot dump tensor of type {}".format(operand.type.type))
    mixedTypeTemplate = """\
{{ // See tools/test_generator/include/TestHarness.h:MixedTyped
  // int -> Dimensions map
  .operandDimensions = {{{dimensions_map}}},
  // int -> FLOAT32 map
  .float32Operands = {{{float32_map}}},
  // int -> INT32 map
  .int32Operands = {{{int32_map}}},
  // int -> QUANT8_ASYMM map
  .quant8AsymmOperands = {{{uint8_map}}},
  // int -> QUANT16_SYMM map
  .quant16SymmOperands = {{{int16_map}}},
  // int -> FLOAT16 map
  .float16Operands = {{{float16_map}}},
  // int -> BOOL8 map
  .bool8Operands = {{{bool8_map}}},
  // int -> QUANT8_SYMM_PER_CHANNEL map
  .quant8ChannelOperands = {{{int8_map}}},
  // int -> QUANT16_ASYMM map
  .quant16AsymmOperands = {{{uint16_map}}},
  // int -> QUANT8_SYMM map
  .quant8SymmOperands = {{{quant8_symm_map}}},
}}"""
    return mixedTypeTemplate.format(
        dimensions_map=tg.GetJointStr(typedMap.get("DIMENSIONS", [])),
        float32_map=tg.GetJointStr(typedMap.get("TENSOR_FLOAT32", [])),
        int32_map=tg.GetJointStr(typedMap.get("TENSOR_INT32", [])),
        uint8_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT8_ASYMM", []) +
                                 typedMap.get("TENSOR_OEM_BYTE", [])),
        int16_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT16_SYMM", [])),
        float16_map=tg.GetJointStr(typedMap.get("TENSOR_FLOAT16", [])),
        int8_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT8_SYMM_PER_CHANNEL", [])),
        bool8_map=tg.GetJointStr(typedMap.get("TENSOR_BOOL8", [])),
        uint16_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT16_ASYMM", [])),
        quant8_symm_map=tg.GetJointStr(typedMap.get("TENSOR_QUANT8_SYMM", []))
    )

# Dump Example file for Cts tests
def DumpCtsExample(example, example_fd):
    print("std::vector<MixedTypedExample>& get_%s() {" % (example.examplesName), file=example_fd)
    print("static std::vector<MixedTypedExample> %s = {" % (example.examplesName), file=example_fd)
    for inputFeedDict, outputFeedDict in example.feedDicts:
        print ('// Begin of an example', file = example_fd)
        print ('{\n.operands = {', file = example_fd)
        inputs = DumpMixedType(example.model.GetInputs(), inputFeedDict)
        outputs = DumpMixedType(example.model.GetOutputs(), outputFeedDict)
        print ('//Input(s)\n%s,' % inputs , file = example_fd)
        print ('//Output(s)\n%s' % outputs, file = example_fd)
        print ('},', file = example_fd)
        if example.expectedMultinomialDistributionTolerance is not None:
          print ('.expectedMultinomialDistributionTolerance = %f' %
                 example.expectedMultinomialDistributionTolerance, file = example_fd)
        print ('}, // End of an example', file = example_fd)
    print("};", file=example_fd)
    print("return %s;" % (example.examplesName), file=example_fd)
    print("};\n", file=example_fd)

# Dump Test file for Cts tests
def DumpCtsTest(example, test_fd):
    testTemplate = """\
TEST_F({test_case_name}, {test_name}) {{
    execute({namespace}::{create_model_name},
            {namespace}::{is_ignored_name},
            {namespace}::get_{examples_name}(){log_file});\n}}\n"""
    if example.model.version is not None:
        testTemplate += """\
TEST_AVAILABLE_SINCE({version}, {test_name}, {namespace}::{create_model_name})\n"""
    print(testTemplate.format(
        test_case_name="DynamicOutputShapeTest" if example.model.hasDynamicOutputShape \
                       else "GeneratedTests",
        test_name=str(example.testName),
        namespace=tg.FileNames.specName,
        create_model_name=str(example.model.createFunctionName),
        is_ignored_name=str(example.model.isIgnoredFunctionName),
        examples_name=str(example.examplesName),
        version=example.model.version,
        log_file=tg.FileNames.logFile), file=test_fd)

# For js
def typeToArray(targetType):
    if targetType in ["INT32", "TENSOR_INT32", "UINT32"]:
        str_array = "Int32Array"
    elif targetType in ["TENSOR_QUANT8_ASYMM"]:
        str_array = "Uint8Array"
    elif targetType in ["TENSOR_QUANT8_SYMM_PER_CHANNEL"]:
        str_array = "Int8Array"
    else :
        str_array = "Float32Array"
    return str_array

def DumpJSTest(model, example, js_fd):
    assert model.compiled
    if model.dumped:
        return

    # check: types
    for t in model.GetTypes():
        if t.type not in Configuration.support_types and \
           t.type not in str(Configuration.support_types).lower():
            print ("    skip not support types: %s (%s)" %(
                   example.examplesName, t.type), file = sys.stderr)
            return
        else :
            # use "TENSOR_FLOAT32" to support "TENSOR_FLOAT16"
            if t.type == "TENSOR_FLOAT16":
                t.type = "TENSOR_FLOAT32"

            # use "FLOAT32" to support "FLOAT16"
            if t.type == "FLOAT16":
                t.type = "FLOAT32"

    '''
    # select 'TENSOR_QUANT8_SYMM_PER_CHANNEL' type models
    per_c_flag = False
    select_types = ["TENSOR_QUANT8_SYMM_PER_CHANNEL"]
    for t in model.GetTypes():
        if t.type in select_types:
            per_c_flag = True

    if per_c_flag == False:
        print ("    skip not select types: %s (%s)" %(
               select_types, example.examplesName), file = sys.stderr)
        return
    '''

    # support layout: NHWC
    for p in example.model.GetParameters():
        if p.type.type == "BOOL":
            if p.GetValueAsNumpy() == False:
                if p in model.operands:
                    model.operands.remove(p)
                for op in model.operations:
                    if p in op.ins:
                        op.ins.remove(p)
            else :
                print ("    skip not support layout: %s (%s)" %(
                       example.examplesName, p.GetValueAsNumpy()), file = sys.stderr)
                return

    # check data type
    for operation in model.operations:
        if operation.optype not in Configuration.check_list.keys() and \
           operation.optype not in str(Configuration.check_list.keys()).lower():
            print ("    skip not support operation code: %s (%s)" %(
                   example.examplesName, operation.optype), file = sys.stderr)
            return
        else :
            for inputIndex in range(len(example.model.GetInputs())):
                t = example.model.GetInputs()[inputIndex].type
                c = Configuration.check_list[operation.optype]["inputs"]
                if inputIndex in c:
                    if t.type not in c[inputIndex]["types"]:
                        print ("    skip not support input(type): %s (%s)" %(
                               example.examplesName, t.type), file = sys.stderr)
                        return
                    if len(t.dimensions) not in c[inputIndex]["dimensions"]:
                        print ("    skip not support input(dimension): %s (%s)" %(
                               example.examplesName, t.dimensions), file = sys.stderr)
                        return
                else :
                    print ("    skip not support input: %s (%s)" %(
                           example.examplesName, example.model.GetInputs()[inputIndex]), file = sys.stderr)
                    return

            for parameterIndex in range(len(example.model.GetParameters())):
                t = example.model.GetParameters()[parameterIndex].type
                c = Configuration.check_list[operation.optype]["inputs"]
                pii = parameterIndex + len(example.model.GetInputs())
                if pii in c:
                    if t.type not in c[pii]["types"]:
                        print ("    skip not support parameter(type): %s (%s)" %(
                               example.examplesName, t.type), file = sys.stderr)
                        return
                    if len(t.dimensions) not in c[pii]["dimensions"]:
                        print ("    skip not support parameter(dimension): %s (%s)" %(
                               example.examplesName, t.dimensions), file = sys.stderr)
                        return
                else :
                    print ("    skip not support parameter: %s (%s)" %(
                           example.examplesName, example.model.GetParameters()[parameterIndex]), file = sys.stderr)
                    return

            for outputIndex in range(len(example.model.GetOutputs())):
                t = example.model.GetOutputs()[outputIndex].type
                c = Configuration.check_list[operation.optype]["outputs"]
                if outputIndex in c:
                    if t.type not in c[outputIndex]["types"]:
                        print ("    skip not support output(type): %s (%s)" %(
                               example.examplesName, t.type), file = sys.stderr)
                        return
                    if len(t.dimensions) not in c[outputIndex]["dimensions"]:
                        print ("    skip not support output(dimension): %s (%s)" %(
                               example.examplesName, t.dimensions), file = sys.stderr)
                        return
                else :
                    print ("    skip not support output: %s (%s)" %(
                           example.examplesName, example.model.GetOutputs()[outputIndex]), file = sys.stderr)
                    return

    # check: input and output and values
    for inputFeedDict, outputFeedDict in example.feedDicts:
        for inputOpName in example.model.GetInputs():
            # check input value is None
            if len(inputFeedDict[inputOpName]) is 0:
                # For "TRANSPOSE": if perm is not given, it is set to (n-1...0)
                if model.operations[0].optype == "TRANSPOSE":
                    perm_value = []
                    perm_dimensions = []
                    for num in range(len(model.operands[0].type.dimensions)):
                        perm_value.insert(0, num)
                    # set "perm" dimensions
                    model.operands[1].type.dimensions.clear()
                    model.operands[1].type.dimensions.append(len(model.operands[0].type.dimensions))
                    # set "perm" value
                    inputFeedDict[inputOpName] = perm_value
                else :
                    print ("    skip input value is None: %s (%s - %s)" %(
                           example.examplesName, model.operations[0].optype, inputOpName), file = sys.stderr)
                    return

    # check: compatible dimensions
    if model.operations[0].optype == "MUL" or model.operations[0].optype == "ADD":
        if model.operands[0].type != model.operands[1].type:
            if len(model.operands[0].type.dimensions) != 1 or len(model.operands[1].type.dimensions) != 1:
                print ("    skip not support input(compatible dimensions): %s (%s - %s)" %(
                       example.examplesName, model.operands[0].type.dimensions,
                       model.operands[1].type.dimensions), file = sys.stderr)
                return

    # check: scale
    if model.operations[0].optype == "CONV_2D" or model.operations[0].optype == "DEPTHWISE_CONV_2D":
        if model.operands[0].type.type == "TENSOR_QUANT8_ASYMM":
            if example.model.GetOutputs()[0].type.scale <= (
               model.operands[0].type.scale * model.operands[1].type.scale):
                print ("    skip not support output(scale): %s (%s <= (%s * %s))" %(
                       example.examplesName, example.model.GetOutputs()[0].type.scale,
                       model.operands[0].type.scale, model.operands[1].type.scale), file = sys.stderr)
                return

    # set js test names
    Configuration.example_count = Configuration.example_count + 1

    test_name = ""
    test_index = ""
    args = "options"
    per_channel_types = dict()
    test_info = tg.FileNames.specName.capitalize().replace("_", " ")
    test_name_array = test_info.split(" ")

    if test_name_array[-1].isdigit():
        if test_name_array[-2] is not None and str(test_name_array[-2]) != "v1":
            test_name = " ".join(test_name_array[:-1])
            test_index = test_name_array[-1]
        else :
            test_name = " ".join(test_name_array[:-1])
            test_name = str(test_name) + "_" + test_name_array[-1]
    else:
        test_name = test_info

    print ("", file = js_fd)

    for inputFeedDict, outputFeedDict in example.feedDicts:
        if Configuration.single_example_flag:
            if test_index == "":
                print ("  it('check result for %s example', async function() {"%test_name, file = js_fd)
            else:
                print ("  it('check result for %s example/%s', async function() {"%(
                       test_name, test_index), file = js_fd)
        else:
            if test_index == "":
                print ("  it('check result for %s example-%s', async function() {"%(
                       test_name, Configuration.example_count), file = js_fd)
            else:
                print ("  it('check result for %s example/%s-%s', async function() {"%(
                       test_name, test_index, Configuration.example_count), file = js_fd)

        print ("    // For '%s' example: %s" %(test_name, example.examplesName), file = js_fd)
        print ("    let model = await nn.createModel(%s);"%args, file = js_fd)
        print ("    let operandIndex = 0;\n", file = js_fd)

        # set input and output values
        for inputOpName in example.model.GetInputs():
            print ("    let %s_value = %s;"%(inputOpName, inputFeedDict[inputOpName]), file = js_fd)
        for outputOpName in example.model.GetOutputs():
            print ("    let %s_expect = %s;"%(outputOpName, outputFeedDict[outputOpName]), file = js_fd)
        print ("", file = js_fd)

        # set input and output types
        for t in model.GetTypes():
            if t.scale == 0.0 and t.zeroPoint == 0 and t.extraParams is None:
                if t.type in ["FLOAT32", "INT32", "UINT32"]:
                    typeDef = "    let %s = {type: nn.%s};"%(t, t.type)
                else :
                    typeDef = "    let %s = {type: nn.%s, dimensions: [%s]};\n    let %s_length = product(%s.dimensions);"%(
                              t, t.type, t.GetDimensionsString()[1:-1], t, t)
            else:
                if t.extraParams is None or t.extraParams.hide:
                    typeDef = "    let %s = {type: nn.%s, dimensions: [%s], scale: %s, zeroPoint: %d};\n    let %s_length = product(%s.dimensions);"%(
                              t, t.type, t.GetDimensionsString()[1:-1], tg.PrettyPrintAsFloat(t.scale)[:-1], t.zeroPoint, t, t)
                else:
                    typeDef = "    let %s = {type: nn.%s, dimensions: [%s]};\n    let %s_length = product(%s.dimensions);"%(
                              t, t.type, t.GetDimensionsString()[1:-1], t, t)

                    per_channel_types[str(t)] = t.extraParams.GetJSConstructor()

            print (typeDef, file = js_fd)
        print ("", file = js_fd)

        # set operands
        for op in model.operands:
            print ("    let %s = operandIndex++;"%op, file = js_fd)
            print ("    model.addOperand(%s);"%op.type, file = js_fd)

            if str(op.type) in per_channel_types.keys():
                print ("    model.setOperandSymmPerChannelQuantParams(operandIndex++, %s);"%per_channel_types[str(op.type)], file = js_fd)
        print ("", file = js_fd)

        # set other inputs value(support only one input)
        if len(example.model.GetInputs()) > 1:
            for inputIndex in range(len(example.model.GetInputs())):
                if inputIndex is not 0:
                    inputType = example.model.GetInputs()[inputIndex].type.type
                    str_array = typeToArray(inputType)
                    print ("    model.setOperandValue(%s, new %s(%s_value));"%(
                           example.model.GetInputs()[inputIndex], str_array,
                           example.model.GetInputs()[inputIndex]), file = js_fd)
            print ("", file = js_fd)

        # set parameter
        for p in model.GetParameters():
            parameterType = p.type.type
            str_array = typeToArray(parameterType)
            print ("    model.setOperandValue(%s, new %s([%s]));"%(
                   p, str_array, GetJointStr(p.value)), file = js_fd)

        # set operations
        for op in model.operations:
            print ("    model.addOperation(nn.%s, [%s], [%s]);"%(
                   op.optype, tg.GetJointStr(op.ins), tg.GetJointStr(op.outs)), file = js_fd)
        print ("", file = js_fd)

        # identify inputs and outputs
        print ("    model.identifyInputsAndOutputs([%s], [%s]);"%(
               example.model.GetInputs()[0], tg.GetJointStr(example.model.GetOutputs())), file = js_fd)
        print ("    await model.finish();", file = js_fd)
        print ("", file = js_fd)

        # compiling model
        print ("    let compilation = await model.createCompilation();", file = js_fd)
        print ("    compilation.setPreference(getPreferenceCode(%s.prefer));"%args, file = js_fd)
        print ("    await compilation.finish();", file = js_fd)
        print ("", file = js_fd)

        # executing model
        print ("    let execution = await compilation.createExecution();", file = js_fd)
        print ("", file = js_fd)

        # set input and output
        inputType = example.model.GetInputs()[0].type.type
        str_array = typeToArray(inputType)
        print ("    let %s_input = new %s(%s_value);"%(
               example.model.GetInputs()[0], str_array, example.model.GetInputs()[0]), file = js_fd)
        print ("    execution.setInput(0, %s_input);"%example.model.GetInputs()[0], file = js_fd)

        for outputIndex in range(len(example.model.GetOutputs())):
            outputType = example.model.GetOutputs()[outputIndex].type.type
            str_array = typeToArray(outputType)
            print ("    let %s_output = new %s(%s_length);"%(
                   example.model.GetOutputs()[outputIndex], str_array,
                   example.model.GetOutputs()[outputIndex].type), file = js_fd)
            print ("    execution.setOutput(%s, %s_output);"%(
                   outputIndex, example.model.GetOutputs()[outputIndex]), file = js_fd)
        print ("", file = js_fd)
        print ("    await execution.startCompute();", file = js_fd)
        print ("", file = js_fd)

        # assert output
        for output in example.model.GetOutputs():
            print ("    for (let i = 0; i < %s_length; ++i) {"%output.type, file = js_fd)
            print ("      assert.isTrue(almostEqualCTS(%s_output[i], %s_expect[i]));"%(output, output), file = js_fd)
            print ("    }", file = js_fd)

        print ("  });", file = js_fd)

    model.dumped = True
# end

if __name__ == '__main__':
    ParseCmdLine()

    while tg.FileNames.NextFile():
        ''' Original
        if Configuration.force_regenerate or NeedRegenerate():
        '''

        # For js
        if Configuration.force_regenerate or NeedRegenerateForJS():
        # end
            print ("--Generating test(s) from spec: %s" %tg.FileNames.specFile, file = sys.stderr)
            exec(open(tg.FileNames.specFile, "r").read())

            ''' Original
            print("Output CTS model: %s" % tg.FileNames.modelFile, file=sys.stderr)
            print("Output example:%s" % tg.FileNames.exampleFile, file=sys.stderr)
            print("Output CTS test: %s" % tg.FileNames.testFile, file=sys.stderr)
            with SmartOpen(tg.FileNames.modelFile) as model_fd, \
                 SmartOpen(tg.FileNames.exampleFile) as example_fd, \
                 SmartOpen(tg.FileNames.testFile) as test_fd:
                InitializeFiles(model_fd, example_fd, test_fd)
                Example.DumpAllExamples(
                    DumpModel=DumpCtsModel, model_fd=model_fd,
                    DumpExample=DumpCtsExample, example_fd=example_fd,
                    DumpTest=DumpCtsTest, test_fd=test_fd)
            '''

            # For js
            with SmartOpen(tg.FileNames.modelFile) as model_fd, \
                 SmartOpen(tg.FileNames.exampleFile) as example_fd, \
                 SmartOpen(tg.FileNames.testFile) as test_fd, \
                 SmartOpen(tg.FileNames.jsFile) as js_fd:
                InitializeFilesForJS(js_fd)
                Example.DumpAllExamples(
                    DumpModel=DumpCtsModel, model_fd=model_fd,
                    DumpExample=DumpCtsExample, example_fd=example_fd,
                    DumpTest=DumpCtsTest, test_fd=test_fd,
                    DumpJS=DumpJSTest, js_fd=js_fd)
                print ("});", file = js_fd)
            # end
        else:
            print ("Skip file: %s" % tg.FileNames.specFile, file = sys.stderr)

        if Configuration.example_count == 0:
            os.remove(tg.FileNames.jsFile)
            print (">>Remove empty JS CTS test: %s\n" %tg.FileNames.jsFile, file = sys.stderr)
        else :
            print (">>Output JS CTS test: %s\n" %tg.FileNames.jsFile, file = sys.stderr)
        ''' Original
        with SmartOpen(tg.FileNames.ctsFile, mode="a") as cts_fd:
            print("#include \"../generated/tests/%s.cpp\""%os.path.basename(tg.FileNames.specFile),
                file=cts_fd)
        '''
