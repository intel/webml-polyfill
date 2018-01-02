import {Enum} from 'enumify';

export class FuseCode extends Enum {}
FuseCode.initEnum(['none', 'relu', 'relu1', 'relu6']);

export class OperandCode extends Enum {}
OperandCode.initEnum(['float32', 'int32', 'uint32', 'tensor_float32', 'tensor_int32', 'tensor_quant8_asymm']);

export class OperationCode extends Enum {}
OperationCode.initEnum(['add', 'average_pool_2d', 'concatenation', 'conv_2d', 'depthwise_conv_2d', 'depth_to_space',
                        'dequantize', 'embedding_lookup', 'floor', 'fully_connected', 'hashtable_lookup',
                        'l2_normalization', 'l2_pool_2d', 'local_response_normalization', 'logistic',
                        'lsh_projection', 'lstm', 'max_pool_2d', 'mul', 'relu', 'relu1', 'relu6', 'reshape',
                        'resize_bilinear', 'rnn', 'softmax', 'space_to_depth', 'svdf', 'tanh']);

export class PaddingCode extends Enum {}
PaddingCode.initEnum(['same', 'valid']);

export class PreferenceCode extends Enum {}
PreferenceCode.initEnum(['low_power', 'fast_single_answer', 'sustained_speed']);

export class OperandLifetime extends Enum {}
OperandLifetime.initEnum(['temporary_variable', 'model_input', 'model_output', 'constant_copy', 'constant_reference', 'no_value']);