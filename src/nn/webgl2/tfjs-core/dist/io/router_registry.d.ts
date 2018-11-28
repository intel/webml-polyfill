import { IOHandler } from './types';
export declare type IORouter = (url: string | string[]) => IOHandler;
export declare class IORouterRegistry {
    private static instance;
    private saveRouters;
    private loadRouters;
    private constructor();
    private static getInstance;
    static registerSaveRouter(saveRouter: IORouter): void;
    static registerLoadRouter(loadRouter: IORouter): void;
    static getSaveHandlers(url: string | string[]): IOHandler[];
    static getLoadHandlers(url: string | string[]): IOHandler[];
    private static getHandlers;
}
