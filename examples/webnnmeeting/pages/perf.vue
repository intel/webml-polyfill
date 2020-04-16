<template>
  <section class="section">
    <div class="home-center control-scale">
      <div class="title">Background Replacement</div>
      <b-table :data="data" :columns="columns"></b-table>
      <div class="ltr fs mt2">
        DeepLab v3+ / MobileNet v2 / Dilated (a.k.a. atrous), ArgMax / 8.4MB /
        Input Size: 224
      </div>
      <div class="title mt">Test Environment</div>
      <div class="columns e">
        <div v-for="e in environment" class="column fs">
          <div class="title">{{ e.model }}</div>
          <div>{{ e.cpu }}</div>
          <div>{{ e.coresthreads }}</div>
          <div>{{ e.baseturbo }}</div>
          <div>{{ e.pcie4 }}</div>
          <div>{{ e.l2cache }}</div>
          <div>{{ e.l3cache }}</div>
          <div>{{ e.tdp }}</div>
          <div>{{ e.mem }}</div>
          <div>{{ e.gpu }}</div>
        </div>
      </div>
    </div>
  </section>
</template>

<script>
export default {
  name: 'Home',
  layout: 'sceneryhome',
  data() {
    return {
      data: [
        {
          backend: 'WASM',
          fps_i10: '1',
          inference_i10: '1235.2',
          fps_9900k: '1',
          inference_9900k: '1235.2',
          fps_3900x: '1',
          inference_3900x: '1235.2'
        },
        {
          backend: 'WebGL',
          fps_i10: '1',
          inference_i10: '1235.2',
          fps_9900k: '1',
          inference_9900k: '1235.2',
          fps_3900x: '1',
          inference_3900x: '1235.2'
        },
        {
          backend: 'WebNN (clDNN)',
          fps_i10: '1',
          inference_i10: '1235.2',
          fps_9900k: '1',
          inference_9900k: '1235.2',
          fps_3900x: '1',
          inference_3900x: '1235.2'
        },
        {
          backend: 'WebNN (DirectML)',
          fps_i10: '1',
          inference_i10: '1235.2',
          fps_9900k: '1',
          inference_9900k: '1235.2',
          fps_3900x: '1',
          inference_3900x: '1235.2'
        },
        {
          backend: 'WebNN (DNNL)',
          fps_i10: '1',
          inference_i10: '1235.2',
          fps_9900k: '1',
          inference_9900k: '1235.2',
          fps_3900x: '1',
          inference_3900x: '1235.2'
        }
      ],
      columns: [
        {
          field: 'backend',
          label: 'Backend'
        },
        {
          field: 'fps_i10',
          label: 'FPS / i10'
        },
        {
          field: 'fps_9900k',
          label: 'FPS / i9-9900K'
        },
        {
          field: 'fps_3900x',
          label: 'FPS / Ryzen 9 3900X'
        },
        {
          field: 'inference_i10',
          label: 'Inference Time (ms) / i10'
        },
        {
          field: 'inference_9900k',
          label: 'Inference Time (ms) / i9-9900K'
        },
        {
          field: 'inference_3900x',
          label: 'Inference Time (ms) / Ryzen 9 3900X'
        }
      ],
      environment: [
        {
          model: 'Model',
          cpu: 'CPU',
          coresthreads: 'Cores / Threads',
          baseturbo: 'Base / Turbo',
          pcie4: 'PCIe 4.0 Lanes',
          l2cache: 'L2 Cache',
          l3cache: 'L3 Cache',
          tdp: 'TDP',
          mem: 'Memory',
          gpu: 'GPU'
        },
        {
          model: 'Ice Lake Laptop',
          cpu: 'Intel i10-',
          coresthreads: '8 / 16',
          baseturbo: '3.6 / 5.0 GHz',
          pcie4: '16 (Free) Gen 3.0',
          l2cache: '256 KB/core',
          l3cache: '16 MB',
          tdp: '95 W',
          mem: '',
          gpu: 'UHD 630 / 0.350-1.20 GHz'
        },
        {
          model: 'Coffee Lake Desktop',
          cpu: 'Intel i9-9900K',
          coresthreads: '8 / 16',
          baseturbo: '3.6 / 5.0 GHz',
          pcie4: '16 (Free) Gen 3.0',
          l2cache: '256 KB/core',
          l3cache: '16 MB',
          tdp: '95 W',
          mem: '',
          gpu: 'UHD 630 / 0.350-1.20 GHz'
        },
        {
          model: 'AMD Zen 2 Desktop',
          cpu: 'AMD Ryzen 9 3900X',
          coresthreads: '12 / 24',
          baseturbo: '3.8 / 4.6 GHz',
          pcie4: '16 (Free) + 4 (NVMe)',
          l2cache: '512 KB/core',
          l3cache: '4x16 MB, 64 MB total',
          tdp: '105 W',
          mem: '',
          gpu: ''
        }
      ]
    }
  }
}
</script>
<style scope>
.home-center {
  margin: 0 auto;
  text-align: center;
}

.title {
  font-size: 1rem;
  color: rgba(255, 255, 255, 1);
  font-weight: 400;
}

.table {
  background-color: transparent;
  color: rgba(255, 255, 255, 0.8);
}

.table td,
.table th {
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-width: 0 0 1px;
  text-align: center;
}

table td:not([align]),
table th:not([align]) {
  text-align: center;
}

.table thead td,
.table thead th {
  color: rgba(204, 255, 144, 1);
  border-width: 0 0 1px;
}

/*
.table th {
  background-color: rgba(0, 0, 0, 0.2);
}

.table tr:nth-child(even) {
  background-color: rgba(0, 0, 0, 0.2);
}
*/

.table:last-child {
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
}

.table tr:hover {
  background-color: rgba(0, 0, 0, 0.2);
  color: rgba(255, 255, 255, 1);
}

.mt {
  margin-top: 1.5rem;
}

.mt2 {
  margin-top: 0.5rem;
}

.columns {
  margin-top: 0;
  padding: 0 1rem;
}

.column {
  border: 0px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.e .column {
  padding: 0;
}

.e .column div {
  border-bottom: 1px solid rgba(255, 255, 255, 0.2);
  padding: 0.25rem 0;
}

.e .column div:hover {
  background-color: rgba(0, 0, 0, 0.2);
  color: rgba(255, 255, 255, 1);
}

.e .column .title {
  font-size: 1rem;
  color: rgba(204, 255, 144, 1);
  padding: 0.5rem 0;
}

.column .title:not(:last-child) {
  margin-bottom: 0rem;
}

.ltr {
  text-align: left;
}

.rtl {
  text-align: right;
}

.fs {
  font-size: 0.8rem;
}

/*

.control-scale:hover,
.control-scale:focus {
  transition: all 0.3s ease;
  transform: scale(1.5);
}

.control-scale:not(:hover) {
  transition: all 1s ease;
  transform: scale(1);
}

@media (max-width: 768px) {
  .control-scale:hover,
  .control-scale:focus {
    transition: all 0.3s ease;
    transform: scale(1.5);
  }
}
*/
</style>
