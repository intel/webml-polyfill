<template>
  <div class="clock">
    <div class="recorder-container">
      <div class="outer"></div>
      <div class="icon-microphone">&nbsp;</div>
    </div>
    <span>{{ clock }}</span>
    <span class="mill">{{ mill }}</span>
  </div>
</template>

<script>
import getTime from '~/assets/js/user/time'
export default {
  data() {
    return {
      clock: null,
      mill: 0
    }
  },
  mounted() {
    setInterval(() => {
      const d = new Date()
      this.clock = getTime() + ':'
      this.mill = d.getMilliseconds()
    }, 1)
  }
}
</script>

<style scope>
.mill {
  width: 24px;
  display: inline-block;
  text-align: left;
  margin-left: -3px;
}

.clock {
  display: inline-block;
}

.recorder-container {
  width: 12px;
  background-color: rgba(204, 255, 144, 1);
  display: inline-block;
  margin-right: 0.75rem;
  border-radius: 100%;
  cursor: default;
  transition: 1s all ease-in;
  position: relative;
}

.icon-microphone {
  color: #fff;
  line-height: 12px;
  display: block;
  text-align: center;
  transition: 0.1s all ease-in;
  position: relative;
}

.outer {
  width: 12px;
  height: 12px;
  margin: 0;
  transform: scale(1);
  border-radius: 100%;
  position: absolute;
  background-color: #fff;
  z-index: -1;
  transition: 1.5s all ease;
  animation: rec 1.5s infinite;
  animation-delay: 1.5s;
}

@keyframes rec {
  0% {
    transform: scale(1.2);
    opacity: 0.5;
  }

  50% {
    transform: scale(1.6);
    opacity: 0.3;
  }

  100% {
    transform: scale(2);
    opacity: 0;
  }
}
</style>
