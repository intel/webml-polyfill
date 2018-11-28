export declare function getReshaped(inputShape: number[], blockShape: number[], prod: number, batchToSpace?: boolean): number[];
export declare function getPermuted(reshapedRank: number, blockShapeRank: number, batchToSpace?: boolean): number[];
export declare function getReshapedPermuted(inputShape: number[], blockShape: number[], prod: number, batchToSpace?: boolean): number[];
export declare function getSliceBeginCoords(crops: number[][], blockShape: number): number[];
export declare function getSliceSize(uncroppedShape: number[], crops: number[][], blockShape: number): number[];
