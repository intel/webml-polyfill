export const fuseShaderSource = {
    "RELU": 'sum = max(sum, 0.0);',
    "RELU1": 'sum = min(max(sum, -1.0), 1.0);',
    "RELU6": 'sum = min(max(sum, 0.0), 6.0);',
    "NONE": ''
}