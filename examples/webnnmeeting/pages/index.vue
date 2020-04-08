<template>
  <section class="section">
    <div class="home-center control-scale">
      <b-field position="is-centered homecontrol">
        <b-input
          v-model="user"
          @keyup.native.enter="join"
          placeholder="type your name"
          type="text"
          icon="account"
        ></b-input>
        <p class="control">
          <button @click="join" class="button join">
            <span class="insider"></span>
            <i class="mdi mdi-near-me mdi-24px"></i>
          </button>
        </p>
      </b-field>
      <div class="settings">
        <b-field v-show="false">
          <template v-for="st in subscribetypes">
            <b-radio-button
              @change.native="updateSubscribeType"
              v-model="subscribetype"
              :native-value="st"
            >
              {{ st }}
            </b-radio-button>
          </template>
        </b-field>
        <b-tabs v-model="enablevideo" @change="updateVideo" class="two">
          <template v-for="videooption in videooptions">
            <b-tab-item :label="videooption"></b-tab-item>
          </template>
        </b-tabs>
        <b-tabs v-model="resolution" @change="updateResolutions" class="three">
          <template v-for="r in resolutions">
            <b-tab-item :label="r"></b-tab-item>
          </template>
        </b-tabs>
        <b-tabs v-model="echocancellation" class="two">
          <template v-for="ec in echocancellations">
            <b-tab-item :label="ec"></b-tab-item>
          </template>
        </b-tabs>
        <b-tabs v-model="noisesuppression" class="two">
          <template v-for="ns in noisesuppressions">
            <b-tab-item :label="ns"></b-tab-item>
          </template>
        </b-tabs>
      </div>
    </div>
    <MeetingInfo />
  </section>
</template>

<script>
import MeetingInfo from '~/components/MeetingInfo.vue'

export default {
  name: 'Home',
  layout: 'sceneryhome',
  components: {
    MeetingInfo
  },
  data() {
    return {
      user: '',
      subscribetype: 'forward',
      subscribetypes: ['forward', 'mix'],
      resolution: 2,
      resolutions: ['320x240', '640x480', '1280x720'],
      enablevideo: 1,
      videooptions: ['Audio Only', 'Video and Audio'],
      echocancellation: 1,
      echocancellations: ['None', 'Echo Cancellation'],
      noisesuppression: 1,
      noisesuppressions: ['None', 'Noise Suppression']
    }
  },
  methods: {
    updateSubscribeType() {
      this.$store.commit('setSubscribeType', this.subscribetype)
    },
    updateVideo() {
      let ev
      this.enablevideo ? (ev = true) : (ev = false)
      this.$store.commit('setEnableVideo', ev)
    },
    updateResolutions() {
      let rswidth, rsheight
      switch (this.resolution) {
        case 0:
          rswidth = 320
          rsheight = 240
          break
        case 1:
          rswidth = 640
          rsheight = 480
          break
        case 2:
          rswidth = 1280
          rsheight = 720
          break
        default:
          rswidth = 1280
          rsheight = 720
      }
      this.$store.commit('setResolutionWidth', rswidth)
      this.$store.commit('setResolutionHeight', rsheight)
    },
    join(e) {
      if (this.user.length <= 0) {
        this.emptyName()
        e.preventDefault()
      }
      if (this.user.length > 0 && this.user.length < 2) {
        this.shortName()
        e.preventDefault()
      }
      if (this.user.length > 48) {
        this.longName()
        e.preventDefault()
      }
      if (this.user.length >= 2 && this.user.length <= 48) {
        // this.$router.push({
        //     path: '/user/' + this.user,
        //     query: {
        //       t: this.subscribetype,
        //       r: this.resolution,
        //       v: this.enablevideo
        //     }
        // })

        const supportwebnn = this.$store.state.supportwebnn
        let bp = null
        if (supportwebnn) {
          bp = '&b=WebML&p=sustained'
        } else {
          bp = '&b=WebGL&p=none'
        }

        const path = '/user/' + this.user
        const query =
          '/?t=' +
          this.subscribetype +
          '&r=' +
          this.resolution +
          '&v=' +
          this.enablevideo +
          '&ec=' +
          this.echocancellation +
          '&ns=' +
          this.noisesuppression +
          bp
        location.href = path + query
      }
    },
    emptyName() {
      this.$buefy.toast.open({
        duration: 3000,
        message: `type name to join the meeting`,
        position: 'is-bottom',
        type: 'is-warning'
      })
    },
    shortName() {
      this.$buefy.toast.open({
        duration: 3000,
        message: `too short name`,
        position: 'is-bottom',
        type: 'is-warning'
      })
    },
    longName() {
      this.$buefy.toast.open({
        duration: 3000,
        message: `too long name`,
        position: 'is-bottom',
        type: 'is-warning'
      })
    }
  }
}
</script>
<style>
.home-center {
  margin: 0 auto;
  text-align: center;
  display: block;
}

.homecontrol {
  margin: 0 auto;
  background-color: transparent !important;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 30px !important;
  height: 60px;
  color: rgba(255, 255, 255, 1);
  width: 360px;
  align-items: center !important;
  padding: 0 8px;
}

.field.has-addons {
  margin-bottom: 0;
}

.settings .b-tabs:not(:last-child) {
  margin-bottom: 0.2rem;
}

.settings {
  border: 1px solid rgba(255, 255, 255, 0.2);
  margin: 0px auto 0 auto;
  padding: 1rem;
  width: 296px;
  border-top: 0px;
}

.settings .field,
.settings .field span,
.settings .field label,
.settings .tabs ul {
  justify-content: center;
  font-size: 0.8rem;
}

.settings .b-radio,
.settings .tabs ul li a {
  border-radius: 0px !important;
  border: 0px;
  height: 24px;
}

.settings .tabs ul {
  margin: 4px 0 0 0;
}

.settings .b-radio {
  background-color: transparent;
  width: 120px;
  text-transform: capitalize;
}

.settings .three .tabs ul li a {
  width: 80px;
}

.settings .two .tabs ul li a {
  width: 120px;
}

.settings .button.is-primary:focus:not(:active),
.settings .button.is-primary.is-focused:not(:active) {
  color: rgba(204, 255, 144, 1) !important;
  box-shadow: 0 0 0 0 rgba(255, 255, 255, 0);
}

.settings .button.is-primary {
  color: rgba(204, 255, 144, 1) !important;
}

.settings .b-radio:hover {
  background-color: rgba(0, 0, 0, 0.2);
  color: rgba(204, 255, 144, 1) !important;
}

.settings .is-focused {
  color: rgba(204, 255, 144, 1) !important;
}

.settings .b-radio,
.settings .tabs ul li a {
  color: rgba(255, 255, 255, 1);
}

.settings .tabs ul li.is-active a {
  color: rgba(204, 255, 144, 1);
}

.settings .tabs ul li:hover a {
  background-color: rgba(0, 0, 0, 0.2);
}

.settings .b-tabs .tab-content {
  display: none;
}

.control-scale:hover,
.control-scale:focus {
  transition: all 0.5s ease;
  transform: scale(1.2);
}

.control-scale:not(:hover) {
  transition: all 1s ease;
  transform: scale(1);
}

@media (max-width: 768px) {
  .control-scale:hover,
  .control-scale:focus {
    transition: all 0.5s ease;
    transform: scale(1.1);
  }
}

.homecontrol:hover,
.homecontrol:focus {
  border: 1px solid rgba(255, 255, 255, 1);
}

.homecontrol .input,
.homecontrol input {
  background-color: transparent !important;
  border: 0px solid transparent !important;
  height: 56px;
  color: rgba(255, 255, 255, 1);
  font-size: 22px;
  font-weight: 300;
  outline: 0 !important;
  box-shadow: 0px 0px 0px rgba(255, 255, 255, 0);
  text-shadow: 1px 1px 1px rgba(0, 0, 0, 0.4);
}

.homecontrol input::placeholder {
  color: rgba(255, 255, 255, 0.8) !important;
}

.homecontrol input::placeholder::after {
  color: rgba(255, 255, 255, 0.8) !important;
}

.homecontrol .control.has-icons-left .icon,
.control.has-icons-right .icon {
  height: 56px;
  color: rgba(255, 255, 255, 0.8);
}

.mdi-24px.mdi-set,
.mdi-24px.mdi:before {
  color: rgba(255, 255, 255, 0.8);
}

.homecontrol .join {
  height: 47px;
  width: 47px;
  border-radius: 24px !important;
  border: 1px solid rgba(204, 255, 144, 1);
  background: transparent;
  color: rgba(255, 255, 255, 0.8);
  border-radius: 24px !important;
  overflow: hidden;
  transform: scale(0.8);
  transition: all 350ms ease-in-out;
}

.join .insider {
  background-color: rgba(255, 255, 255, 0.8);
  width: 10px;
  height: 100px;
  position: absolute;
  left: -135px;
  transform: rotateZ(135deg);
}

.join:hover {
  border-color: rgba(204, 255, 144, 1);
  color: #fff;
  transform: scale(1);
}

.join:hover .insider {
  transition: all 1s ease;
  left: 135px;
}
</style>
