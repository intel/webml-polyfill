import { CompilationOptions } from "./CompilationOptions";
import { Execution } from "./Execution";
import { Model } from "./Model";

export class Compilation {
  private model_: Model;

  constructor(options: CompilationOptions, model: Model) {
    this.model_ = model;
  }

  async createExecution(): Promise<Execution> {
    return new Execution(this.model_);
  }
}