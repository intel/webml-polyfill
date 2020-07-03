import { Pool } from './Pool'

export class AveragePool2d extends Pool {
  getPoolingType(): 'avg'|'max' {
    return 'avg';
  }
}