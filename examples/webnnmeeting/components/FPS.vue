<template>
  <span id="fps"></span>
</template>
<script>
export default {
  name: 'FPS',
  head: {
    script: [{ src: '../js/fps.js', defer: true }]
  },
  data() {
    return {
      test: ''
    }
  }
}
</script>
<style scope></style>
