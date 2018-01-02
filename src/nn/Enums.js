import {Enum} from 'enumify';

export class FuseCode extends Enum {}
FuseCode.initEnum(['none', 'relu', 'relu1', 'relu6']);

export class OperandCode extends Enum {}
OperandCode.initEnum(['float32', 'int32', 'uint32', 'tensor-float32', 'tensor-int32', 'tensor-quant8-asymm']);

export class OperationCode extends Enum {}
OperationCode.initEnum(['add', 'average-pool-2d', 'concatenation', 'conv-2d', 'depthwise-conv-2d', 'depth-to-space',
                        'dequantize', 'embedding-lookup', 'floor', 'fully-connected', 'hashtable-lookup',
                        'l2-normalization', 'l2-pool-2d', 'local-response-normalization', 'logistic',
                        'lsh-projection', 'lstm', 'max-pool-2d', 'mul', 'relu', 'relu1', 'relu6', 'reshape',
                        'resize-bilinear', 'rnn', 'softmax', 'space-to-depth', 'svdf', 'tanh']);

export class PaddingCode extends Enum {}
PaddingCode.initEnum(['same', 'valid']);

export class PreferenceCode extends Enum {}
PreferenceCode.initEnum(['low-power', 'fast-single-answer', 'sustained-speed']);
