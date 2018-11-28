export declare type ShapeInfo = {
    logicalShape: number[];
    texShape: [number, number];
    isUniform: boolean;
    isPacked: boolean;
};
export declare type InputInfo = {
    name: string;
    shapeInfo: ShapeInfo;
};
export declare function makeShader(inputsInfo: InputInfo[], outputShape: ShapeInfo, userCode: string, broadcast: boolean, usesPackedTextures: boolean): string;
export declare function getCoordsDataType(rank: number): string;
