<template>
  <div :class="getFullscreen ? 'fullscreen' : ''" class="hero-foot">
    <div class="container has-text-centered section">
      <div class="footerbrackets brackets">
        <a
          href="https://intel.github.io/webml-polyfill/examples/"
          title="Web Neural Network API Examples"
          >Examples</a
        >
        <a
          href="https://webmachinelearning.github.io/webnn"
          title="W3C Web Neural Network API Specification"
          >W3C Spec</a
        >
        <a
          href="https://www.w3.org/community/webmachinelearning/"
          title="W3C Machine Learning for the Web Community Group"
          >W3C CG</a
        >
        <!-- <a href="../../perf" title="WebNN Meeting Performance">Performance</a> -->
      </div>
      <div id="copyright">
        &copy;2020
        <a
          href="https://github.com/intel/webml-polyfill"
          title="Web Neural Network API"
          >WebNN API</a
        >
      </div>
      <WebNNBadge />
    </div>
  </div>
</template>

<script>
import WebNNBadge from '~/components/WebNNBadge.vue'

export default {
  components: {
    WebNNBadge
  },
  computed: {
    getFullscreen() {
      return this.$store.state.fullscreen
    }
  }
}
</script>

<style scope>
.hero-foot .section {
  padding: 0 0 1rem 0 !important;
}

.hero-foot.fullscreen .brackets,
.hero-foot.fullscreen #copyright {
  display: none;
}

.hero-foot .webnnbadge {
  position: absolute;
  right: 0px;
  bottom: 22px;
}

.fullscreen {
  margin: 0px auto;
}

.fullscreen .webnnbadge {
  right: unset;
  position: unset;
  display: block;
  margin: -90px 0 0 -60px;
}

.hero-foot.fullscreen {
  display: none;
}
</style>
