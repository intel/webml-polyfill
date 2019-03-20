export default class Graph {

  constructor(vertices) {
    this.vertices = vertices;
    this.color = new Array(vertices).fill(false); // false - white, true - black
    this.next = [];
    this.prev = [];
    this.tensors = new Map();
    this.tensorMapping = [];
    this.inTensorsOfInputNode = new Map();
    this.outTensorsOfOutputNode = new Map();
    for (let i = 0; i < vertices; i++) {
      this.next[i] = [];
      this.prev[i] = [];
      this.tensorMapping[i] = [];
    }
  }

  addEdge(i, j, tensor) {
    // at most one tensor attached to edge i->j
    this.next[i].push(j);
    this.prev[j].push(i);
    this.tensorMapping[i][j] = typeof tensor !== 'undefined' ? tensor : -1;
  }

  addNode(nodeId, inTensors, outTensors) {
    for (const i of inTensors) {
      if (!this.tensors.has(i)) {
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
      if (!this.tensors.has(i)) {
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

  setBlack(i) {
    this.color[i] = true;
  }

  identifyInputOutputTensors(inTensors, outTensors) {
    for (const t of inTensors) {
      if (!this.tensors.has(t)) {
        return;
      }
      for (const n of this.tensors.get(t).to) {
        if (!this.inTensorsOfInputNode.has(n)) {
          this.inTensorsOfInputNode.set(n, new Set());
        }
        this.inTensorsOfInputNode.get(n).add(t);
      }
    }
    for (const t of outTensors) {
      if (!this.tensors.has(t)) {
        return;
      }
      for (const n of this.tensors.get(t).from) {
        if (!this.outTensorsOfOutputNode.has(n)) {
          this.outTensorsOfOutputNode.set(n, new Set());
        }
        this.outTensorsOfOutputNode.get(n).add(t);
      }
    }
  }

  topologicalSort() {
    const indegree = new Array(this.vertices).fill(0);
    const result = [];
    const q = [];
    for (let i = 0; i < this.vertices; i++) {
      indegree[i] = this.prev[i].length;
      if (!indegree[i]) {
        q.push(i); // push node i with indegree zero
      }
    }
    let cnt = 0;
    while (q.length) {
      const u = q.shift();
      result.push(u);
      cnt++;
      for (const v of this.next[u]) {
        if (!--indegree[v]) {
          q.push(v);
        }
      }
    }
    if (cnt !== this.vertices) {
      throw new Error('Not a DAG');
    }
    return result;
  }

  biTopologicalSort() {
    const order = new Array(this.vertices).fill(0);
    for (const u of this.topologicalSort()) {
      for (const v of this.prev[u]) {
        if (this.color[u] === this.color[v]) {
          order[u] = Math.max(order[u], order[v]);
        } else {
          order[u] = Math.max(order[u], order[v] + 1);
        }
      }
    }
    const result = [];
    for (const [nodeId, ord] of order.entries()) {
      if (typeof result[ord] === 'undefined') {
        result[ord] = new Set();
      }
      result[ord].add(nodeId);
    }
    return result;
  }

  partition(eager = false) {
    function union(a, b) {
      return new Set([...a, ...b]);
    }

    function sortSet(set) {
      return Array.from(set).sort((a, b) => a - b);
    }
    const result = [];
    // crossTensor - tensor lies on the cross edge
    const crossTensorsTo = new Map();
    for (let i = 0; i < this.vertices; i++) {
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
        for (const v of this.next[u]) {
          if (!partition.has(v)) {
            const tensorUV = this.tensorMapping[u][v];
            crossTensorsTo.get(v).add(tensorUV);
            outTensors.add(tensorUV);
          }
        }
        if (this.outTensorsOfOutputNode.has(u)) {
          outTensors = union(outTensors, this.outTensorsOfOutputNode.get(u));
        }
      }
      for (const u of partition) {
        inTensors = union(inTensors, crossTensorsTo.get(u));
        if (this.inTensorsOfInputNode.has(u)) {
          inTensors = union(inTensors, this.inTensorsOfInputNode.get(u));
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