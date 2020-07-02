<template>
  <section class="hero is-fullheight home content--canvas">
    <Nav />
    <div class="hero-body">
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
  head() {
    return {
      script: [
        { src: '../js/socket.io.js', defer: true },
        { src: '../js/adapter-7.0.0.js', defer: true },
        { src: '../js/dynamic/noise.min.js', defer: true },
        { src: '../js/dynamic/shift.js', defer: true }
      ]
    }
  },
  components: {
    Nav,
    Footer
  },
  method: {}
}
</script>
<style scope>
.content--canvas canvas {
  z-index: -1000;
}
</style>
