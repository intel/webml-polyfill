<template>
  <section
    :class="getFullscreen ? 'isfullscreen' : ''"
    class="hero is-fullheight home"
  >
    <Nav />
    <div id="slider" :data-images="slideimage"></div>
    <div id="hero-body" class="hero-body">
      <div class="container has-text-centered">
        <nuxt />
      </div>
    </div>
    <Footer />
  </section>
</template>

<script>
import Nav from '~/components/Nav.vue'
import Footer from '~/components/Footer.vue'

export default {
  head: {
    script: [
      { src: '../../js/socket.io.js', defer: true },
      { src: '../../js/adapter-7.0.0.js', defer: true },
      { src: '../../js/three.js', defer: true },
      { src: '../../js/gsap.js', defer: true },
      { src: '../../js/sketch.js', defer: true },
      {
        src: '../../js/webnn/polyfill/webml-polyfill.js',
        defer: true
      },
      {
        src: '../../js/webnn/util/base.js',
        defer: true
      },
      {
        src: '../../js/webnn/util/tflite/flatbuffers/js/flatbuffers.js',
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
  components: {
    Nav,
    Footer
  },
  data() {
    return {
      slideimage: [
        '../../img/01.jpg',
        '../../img/02.jpg',
        '../../img/03.jpg',
        '../../img/04.jpg'
      ]
    }
  },
  computed: {
    getFullscreen() {
      return this.$store.state.fullscreen
    }
  },
  methods: {}
}
</script>
<style scope>
.home {
  color: rgba(255, 255, 255, 1);
}

.hero-body {
  padding: 0;
}
</style>
