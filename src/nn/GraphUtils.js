export default class Graph {

  constructor(vertices) {
    this.vertices = vertices;
    this.color = new Array(vertices).fill(false); // false - white, true - black
    this.tensors = new Map();
    this.next = [];
    this.inTensorsOfInputNode = new Map();
    this.outTensorsOfOutputNode = new Map();
    for (let i = 0; i < vertices; i++) {
      this.next[i] = new Map();
    }
  }

  addEdge(i, j, tensor) {
    this.next[i].set(j, tensor); // at most one tensor attached to edge i->j
  }

  addNode(nodeId, inTensors, outTensors) {
    nodeId = parseInt(nodeId);
    for (const i of inTensors) {
      if (typeof this.tensors.get(i) === 'undefined') {
        this.tensors.set(i, {
          from: new Set(),
          to: new Set()
        });
      }
      for (const inNodeId of this.tensors.get(i).from) {
        this.addEdge(inNodeId, nodeId, i);
      }
      this.tensors.get(i).to.add(nodeId);
    }
    for (const i of outTensors) {
      if (typeof this.tensors.get(i) === 'undefined') {
        this.tensors.set(i, {
          from: new Set(),
          to: new Set()
        });
      }
      for (const outNodeId of this.tensors.get(i).to) {
        this.addEdge(nodeId, outNodeId, i);
      }
      this.tensors.get(i).from.add(nodeId);
    }
  }

  identifyInputOutputTensors(inTensors, outTensors) {
    for (const t of inTensors) {
      for (const n of this.tensors.get(t).to) {
        if (typeof this.inTensorsOfInputNode.get(n) === 'undefined') {
          this.inTensorsOfInputNode.set(n, new Set());
        }
        this.inTensorsOfInputNode.get(n).add(t);
      }
    }
    for (const t of outTensors) {
      for (const n of this.tensors.get(t).from) {
        if (typeof this.outTensorsOfOutputNode.get(n) === 'undefined') {
          this.outTensorsOfOutputNode.set(n, new Set());
        }
        this.outTensorsOfOutputNode.get(n).add(t);
      }
    }
  }

  topologicalSort() {
    const _this = this;
    const visited = new Array(this.vertices).fill(false);
    const result = [];
    const dfs = (v) => {
      visited[v] = true;
      for (const i of _this.next[v].keys()) {
        if (!visited[i]) {
          dfs(i);
        }
      }
      result.unshift(v);
    };
    for (let i = 0; i < this.vertices; i++) {
      if (!visited[i]) {
        dfs(i);
      }
    }
    return result;
  }

  setBlack(i) {
    this.color[i] = true;
  }

  biTopologicalSort() {
    const _this = this;

    function dfsColor(v, color) {
      const result = new Set();
      const visited = new Array(_this.vertices).fill(false);
      const _dfsColor = (v) => {
        visited[v] = true;
        if (_this.color[v] !== color) {
          return;
        }
        result.add(v);
        for (const i of _this.next[v].keys()) {
          if (!visited[i]) {
            _dfsColor(i);
          }
        }
      };
      for (const i of _this.next[v].keys()) {
        if (!visited[i]) {
          _dfsColor(i);
        }
      }
      return result;
    }

    function dfsSameColor(v) {
      return dfsColor(v, _this.color[v]);
    }

    function dfsDiffColor(v) {
      return dfsColor(v, !_this.color[v]);
    }

    function diff(a, b) {
      return new Set([...a].filter(x => !b.has(x)));
    }

    function union(a, b) {
      return new Set([...a, ...b]);
    }

    const result = [];
    const processed = new Array(this.vertices).fill(false);
    const topoOrder = this.topologicalSort();
    const topoIndex = new Array(topoOrder.length);
    for (const [i, v] of topoOrder.entries()) {
      topoIndex[v] = i;
    }

    for (const i of topoOrder) {
      if (processed[i])
        continue;
      const sameColorSet = dfsSameColor(i);
      let partition = sameColorSet.add(i);
      // console.log(`processing node ${i}`);
      let maxTopoIndexInPartition = Number.MIN_SAFE_INTEGER;
      let minTopoIndexInPartition = Number.MAX_SAFE_INTEGER;
      for (const v of partition) {
        if (topoIndex[v] > maxTopoIndexInPartition) {
          maxTopoIndexInPartition = topoIndex[v];
        }
        if (topoIndex[v] < minTopoIndexInPartition) {
          minTopoIndexInPartition = topoIndex[v];
        }
      }
      const currColor = _this.color[i];
      for (let j = minTopoIndexInPartition; j <= maxTopoIndexInPartition; j++) {
        const node = topoOrder[j];
        if (!processed[node] && _this.color[node] !== currColor) {
          const diffColorSet = dfsDiffColor(node);
          partition = diff(partition, diffColorSet);
        }
      }
      partition.forEach((v) => processed[v] = true);
      result.push(partition);
    }
    // merge adjacent partitions with same color
    const merged = [result.shift()];
    while (result.length) {
      const prev = merged.pop();
      const curr = result.shift();
      const partitionColor = (s) => _this.color[Array.from(s)[0]];
      if (partitionColor(prev) === partitionColor(curr)) {
        merged.push(union(prev, curr));
      } else {
        merged.push(prev);
        merged.push(curr);
      }
    }
    return merged;
  }
  partition(eager = false) {
    const _this = this;

    function union(a, b) {
      return new Set([...a, ...b]);
    }

    function sortSet(set) {
      return Array.from(set).sort((a, b) => a - b);
    }
    const result = [];
    // crossTensor - tensor lies on the cross edge
    const crossTensorsTo = new Map();
    for (let i = 0; i < _this.vertices; i++) {
      crossTensorsTo.set(i, new Set());
    }

    let partitions = [];
    if (eager) {
      for (const i of this.topologicalSort())
        partitions.push(new Set([i]));
    } else {
      partitions = this.biTopologicalSort();
    }

    for (const partition of partitions) {
      let inTensors = new Set();
      let outTensors = new Set();
      for (const u of partition) {
        for (const v of _this.next[u].keys()) {
          if (!partition.has(v)) {
            const tensorUV = _this.next[u].get(v);
            crossTensorsTo.get(v).add(tensorUV);
            outTensors.add(tensorUV);
          }
        }
        if (_this.outTensorsOfOutputNode.has(u)) {
          outTensors = union(outTensors, _this.outTensorsOfOutputNode.get(u));
        }
      }
      for (const u of partition) {
        inTensors = union(inTensors, crossTensorsTo.get(u));
        if (_this.inTensorsOfInputNode.has(u)) {
          inTensors = union(inTensors, _this.inTensorsOfInputNode.get(u));
        }
      }
      result.push({
        nodes: sortSet(partition),
        inTensors: sortSet(inTensors),
        outTensors: sortSet(outTensors),
      });
    }
    return result;
  }
}