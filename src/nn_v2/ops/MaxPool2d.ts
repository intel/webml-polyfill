import { Pool } from './Pool'

export class MaxPool2d extends Pool {
  getPoolingType(): 'avg'|'max' {
    return 'max';
  }
}