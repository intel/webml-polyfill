<template>
  <div class="webnnbadge">
    <b-tooltip
      v-if="webmlstatus"
      type="is-dark"
      position="is-left"
      square
      label="Your browser supports Web Neural Network API."
    >
      <div class="btn webnn-supported">
        <span>WebNN API</span><b-icon icon="emoticon-happy"></b-icon>
      </div>
    </b-tooltip>
    <b-tooltip
      v-else
      type="is-dark"
      position="is-left"
      square
      label="Your browser does not support Web Neural Network API"
    >
      <div class="btn webnn-not-supported">
        <span>WebNN API</span><b-icon icon="emoticon-sad"></b-icon>
      </div>
    </b-tooltip>
  </div>
</template>
<script>
export default {
  name: 'WebNNBadge',
  data() {
    return {
      webmlstatus: false
    }
  },
  mounted() {
    setTimeout(this.updateWebNNStatus, 1000)
  },
  methods: {
    updateWebNNStatus() {
      if (navigator.ml && navigator.ml.getNeuralNetworkContext()) {
        if (!navigator.ml.isPolyfill) {
          this.webmlstatus = true
          this.$store.commit('setWebNN', true)
        } else {
          this.webmlstatus = false
          this.$store.commit('setWebNN', false)
        }
      } else {
        this.webmlstatus = false
        this.$store.commit('setWebNN', false)
      }
    }
  },
  destoryed() {
    clearTimeout(this.updateWebMLStatus)
  }
}
</script>
<style>
.webnnbadge {
  display: inline-block;
  font-size: 0.8rem;
}

.btn {
  outline: 0;
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  width: 132px;
  border: 0;
  border-top-right-radius: 20px;
  border-bottom-right-radius: 20px;

  /* 
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    border-radius: 4px;
  */
  box-sizing: border-box;
  padding: 4px 10px;
  color: #ffffff;
  font-size: 12px;
  letter-spacing: 1.2px;
  text-transform: uppercase;
  overflow: hidden;
  cursor: pointer;
}

.btn .icon {
  padding-left: 1px;
}

.webnn-supported {
  background: rgba(83, 128, 247, 0.05);
}

.webnn-not-supported {
  background: rgba(255, 71, 15, 0.05);
}

.webnn-supported:hover {
  background: rgba(83, 128, 247, 0.8);
}

.webnn-not-supported:hover {
  background: rgba(255, 71, 15, 0.8);
}

.btn .icon {
  background-color: transparent;
  border-radius: 100%;
  animation: ripple 0.6s linear infinite;
}
</style>
