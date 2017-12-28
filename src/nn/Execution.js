class Execution {
  constructor(compilation) {}

  release() {}

  setInput(index, buffer, type) {}

  setInputFromMemory(index, memory, offset, length, type) {}

  setOutput(index, buffer, type) {}

  setOutputFromMemory(index, memory, offset, length, type) {}

  async compute() {}
}