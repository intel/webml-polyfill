<template>
  <div
    :class="this.$parent.isFullscreen ? 'fullscreencontrolbar' : ''"
    class="meetingcontrol"
  >
    <transition name="fade-slide">
      <div
        v-if="controlbar"
        v-show="this.$parent.initss && showaimenu && !this.$parent.isPauseVideo"
        class="meetingcontrolai"
      >
        <b-button
          @click="ss('blur')"
          v-if="this.$parent.blurdone"
          icon-left="blur"
          type="is-twitter"
          class="btneffect"
        >
          Blur background
        </b-button>
        <b-button v-else @click="ss('blur')" icon-left="blur" type="is-twitter"
          >Blur background
        </b-button>

        <b-progress
          v-if="this.$parent.isblur && this.$parent.blurdone"
          :value="this.$parent.progress"
          type="is-twitter"
          class="nnprogress"
        >
        </b-progress>
        <b-progress
          v-else-if="this.$parent.isblur"
          :value="this.$parent.progress"
          type="is-danger"
          class="nnprogress"
        >
        </b-progress>

        <b-button
          @click="ss('image')"
          v-if="this.$parent.bgimgdone"
          icon-left="image-multiple"
          type="is-twitter"
          class="btneffect"
        >
          Change background
        </b-button>
        <b-button
          @click="ss('image')"
          v-else
          icon-left="image-multiple"
          type="is-twitter"
          >Change background
        </b-button>

        <b-progress
          v-if="this.$parent.isbgimg && this.$parent.bgimgdone"
          :value="this.$parent.progress"
          type="is-twitter"
          class="nnprogress"
        >
        </b-progress>
        <b-progress
          v-else-if="this.$parent.isbgimg"
          :value="this.$parent.progress"
          type="is-danger"
          class="nnprogress"
        >
        </b-progress>
      </div>
    </transition>
    <div>
      <div class="togglecontrolbar">
        <b-button
          v-if="!controlbar"
          @click="toggleControlBar"
          icon-left="chevron-right"
        ></b-button>
        <b-button
          v-if="this.$parent.isFullscreen && !controlbar"
          @click="this.$parent.exitFullScreen"
          icon-left="fullscreen-exit"
        ></b-button>
      </div>
      <transition name="slide-fade">
        <div v-if="controlbar">
          <b-button
            @click="toggleControlBar"
            icon-left="chevron-left"
          ></b-button>
          <b-button
            v-if="this.$parent.isFullscreen"
            @click="this.$parent.exitFullScreen"
            icon-left="fullscreen-exit"
          ></b-button>

          <!-- <b-button class="date"><Clock /></b-button> -->

          <b-button
            v-if="this.$parent.isPauseVideo"
            @click="this.$parent.toggleVideo"
            icon-left="video-off"
          ></b-button>
          <b-button
            v-else
            @click="this.$parent.toggleVideo"
            icon-left="video"
          ></b-button>
          <b-button
            v-if="!this.$parent.isPauseAudio"
            @click="this.$parent.toggleAudio"
            icon-left="microphone"
          ></b-button>
          <b-button
            v-else
            @click="this.$parent.toggleAudio"
            icon-left="microphone-off"
          ></b-button>

          <b-button
            v-if="this.$parent.isLocalScreenSharing"
            @click="this.$parent.shareScreen"
            icon-left="projector-screen"
            class="btnactive"
          ></b-button>
          <b-button
            v-else
            @click="this.$parent.shareScreen"
            icon-left="projector-screen"
          ></b-button>

          <b-button
            v-if="
              this.$parent.initss && showaimenu && !this.$parent.isPauseVideo
            "
            @click="showAiMenu"
            icon-left="dots-horizontal"
            class="btnactive"
          ></b-button>
          <b-button
            v-else-if="this.$parent.initss && !this.$parent.isPauseVideo"
            @click="showAiMenu"
            icon-left="dots-horizontal"
          ></b-button>

          <b-button
            v-if="this.$parent.showconversation"
            @click="toggleConversation"
            class="btnactive"
            icon-left="message-reply-text"
          ></b-button>
          <b-button
            v-else
            @click="toggleConversation"
            icon-left="message-reply-text"
          ></b-button>

          <b-button
            @click="toggleParticipants"
            v-if="this.$parent.showparticipants"
            class="btnactive"
            icon-left="account-group"
          ></b-button>
          <b-button
            @click="toggleParticipants"
            v-else
            icon-left="account-group"
          ></b-button>
          <b-button
            @click="leaveMeeting"
            icon-left="phone-hangup"
            class="leavemeeting"
          ></b-button>
        </div>
      </transition>
    </div>
  </div>
</template>

<script>
// import Clock from '~/components/Clock.vue'

export default {
  components: {
    // Clock
  },
  data() {
    return {
      showaimenu: false,
      controlbar: true
    }
  },
  methods: {
    ss(effect) {
      if (effect === 'blur') {
        this.$parent.isblur = !this.$parent.isblur
        this.$parent.isbgimg = false
        this.$parent.showrightsidebar = false
        if (this.$parent.isblur) {
          this.$parent.ss(effect)
        } else {
          this.$parent.stopSS()
        }
      }

      if (effect === 'image') {
        this.$parent.isbgimg = !this.$parent.isbgimg
        this.$parent.isblur = false
        if (this.$parent.isbgimg) {
          this.$parent.showrightsidebar = true
        } else {
          this.$parent.showrightsidebar = false
        }
        if (this.$parent.isbgimg) {
          this.$parent.ss(effect)
        } else {
          this.$parent.stopSS()
        }
      }
    },
    leaveMeeting() {
      this.$parent.leaveMeeting()
    },
    showAiMenu() {
      this.showaimenu = !this.showaimenu
    },
    toggleParticipants() {
      this.$parent.showparticipants = !this.$parent.showparticipants
    },
    toggleConversation() {
      this.$parent.showconversation = !this.$parent.showconversation
    },
    toggleControlBar() {
      this.controlbar = !this.controlbar
    }
  }
}
</script>

<style scope>
.progress-wrapper:not(:last-child) {
  margin-bottom: 0;
}

.nnprogress .progress {
  height: 1px;
  border-radius: 0px;
  background: transparent;
  border: 0px;
}

.progress-wrapper .progress-value {
  display: block;
  height: 100px;
}

.sseffect {
  width: 120px;
  display: inline-block;
  text-align: left;
  color: rgba(204, 255, 144, 1);
}

.btneffect {
  color: rgba(204, 255, 144, 1);
}

.c {
  position: relative;
  left: 26px;
  top: 3px;
}

.m {
  position: relative;
  left: 8px;
  top: 3px;
}
</style>
