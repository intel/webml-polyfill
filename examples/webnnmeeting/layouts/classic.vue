<template>
  <nuxt />
</template>

<script>
export default {
  head: {
    script: [
      { src: '../../js/socket.io.js', defer: true },
      { src: '../../js/adapter-7.0.0.js', defer: true },
      {
        src: '../../js/webnn/polyfill/webml-polyfill.js',
        defer: true
      },
      {
        src: '../../js/webnn/util/base.js',
        defer: true
      },
      {
        src: '../../js/webnn/util/tflite/flatbuffers.js',
        defer: true
      },
      {
        src: '../../js/webnn/util/tflite/schema/schema_generated.js',
        defer: true
      },
      {
        src: '../../js/webnn/util/tflite/TfLiteModelUtils.js',
        defer: true
      },
      {
        src: '../../js/webnn/util/tflite/TFliteModelImporter.js',
        defer: true
      },
      {
        src: '../../js/webnn/webgl/WebGLUtils.js',
        defer: true
      },
      {
        src: '../../js/webnn/webgl/Shader.js',
        defer: true
      },
      {
        src: '../../js/webnn/webgl/GuidedFilter.js',
        defer: true
      }
    ]
  },
  components: {},
  data() {
    return {
      test: ''
    }
  }
}
</script>
<style scope>
body {
  -ms-overflow-style: none;
  overflow: -moz-scrollbars-none;
  text-shadow: 0px 0px 0px rgba(0, 0, 0, 0) !important;
  color: #333;
}

body::-webkit-scrollbar {
  display: none;
}

body::scrollbar {
  display: none;
}

.column {
  border: 1px solid rgba(255, 255, 255, 0.2);
}
</style>
