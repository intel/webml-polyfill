export interface Features {
    'DEBUG'?: boolean;
    'IS_BROWSER'?: boolean;
    'IS_NODE'?: boolean;
    'WEBGL_LAZILY_UNPACK'?: boolean;
    'WEBGL_CPU_FORWARD'?: boolean;
    'WEBGL_PACK_BATCHNORMALIZATION'?: boolean;
    'WEBGL_CONV_IM2COL'?: boolean;
    'WEBGL_PAGING_ENABLED'?: boolean;
    'WEBGL_MAX_TEXTURE_SIZE'?: number;
    'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'?: number;
    'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'?: boolean;
    'WEBGL_VERSION'?: number;
    'HAS_WEBGL'?: boolean;
    'WEBGL_RENDER_FLOAT32_ENABLED'?: boolean;
    'WEBGL_DOWNLOAD_FLOAT_ENABLED'?: boolean;
    'WEBGL_FENCE_API_ENABLED'?: boolean;
    'WEBGL_SIZE_UPLOAD_UNIFORM'?: number;
    'BACKEND'?: string;
    'TEST_EPSILON'?: number;
    'IS_CHROME'?: boolean;
    'IS_TEST'?: boolean;
    'EPSILON'?: number;
    'PROD'?: boolean;
    'TENSORLIKE_CHECK_SHAPE_CONSISTENCY'?: boolean;
}
export declare enum Type {
    NUMBER = 0,
    BOOLEAN = 1,
    STRING = 2
}
export declare const URL_PROPERTIES: URLProperty[];
export interface URLProperty {
    name: keyof Features;
    type: Type;
}
export declare function isWebGLVersionEnabled(webGLVersion: 1 | 2): boolean;
export declare function getWebGLMaxTextureSize(webGLVersion: number): number;
export declare function getWebGLDisjointQueryTimerVersion(webGLVersion: number): number;
export declare function isRenderToFloatTextureEnabled(webGLVersion: number): boolean;
export declare function isDownloadFloatTextureEnabled(webGLVersion: number): boolean;
export declare function isWebGLFenceEnabled(webGLVersion: number): boolean;
export declare function isChrome(): boolean;
export declare function getFeaturesFromURL(): Features;
export declare function getQueryParams(queryString: string): {
    [key: string]: string;
};
