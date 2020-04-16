<template>
  <section class="hero is-fullheight home">
    <Nav />
    <div id="shadow"></div>
    <div id="slider" :data-images="slideimage"></div>
    <!-- Hero content: will be in the middle -->
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
      { src: '../js/three.js', defer: true },
      { src: '../js/gsap.js', defer: true },
      { src: '../js/sketch.js', defer: true }
    ]
  },
  components: {
    Nav,
    Footer
  },
  data() {
    return {
      slideimage: [
        '../img/01.jpg',
        '../img/02.jpg',
        '../img/03.jpg',
        '../img/04.jpg'
      ]
    }
  }
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
