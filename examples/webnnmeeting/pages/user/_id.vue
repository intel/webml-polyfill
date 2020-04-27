<template>
  <div>
    <div class="columns user">
      <transition name="fade-slide">
        <div
          v-show="showparticipants || showconversation"
          class="column cl nopadding is-one-fifth"
        >
          <div id="layoutparticipants" v-show="showparticipants">
            <div class="isleft pd">PARTICIPANTS</div>
            <div class="isleft pd2">
              Presenters ({{ this.$store.state.participants.number }})
            </div>

            <div ref="userlist" class="userlist">
              <div v-for="u in users" class="columns">
                <div class="column ull isleft is-three-quarters">
                  <b-icon class="ulicon" icon="account" size="is-small">
                  </b-icon>
                  <span class="ulu">{{ u.userId }}</span>
                </div>
                <div class="column ulr">
                  <b-icon
                    v-if="
                      (u.shareScreenStream && isScreenSharing) ||
                        (isLocalScreenSharing && u.userId === localuser.userId)
                    "
                    icon="projector-screen"
                    size="is-small"
                  >
                  </b-icon>
                  <b-icon v-if="u.video" icon="video" size="is-small"> </b-icon>
                  <b-icon v-else icon="video-off" size="is-small"> </b-icon>
                  <b-icon v-if="u.muted" icon="microphone-off" size="is-small">
                  </b-icon>
                  <b-icon v-else icon="microphone" size="is-small"> </b-icon>
                </div>
              </div>
            </div>
          </div>
          <div
            v-show="showparticipants && showconversation"
            class="issplit"
          ></div>
          <div id="layoutconversation" v-show="showconversation">
            <div class="isleft pd">CONVERSATION</div>
            <div ref="conversation" class="conversation">
              <div v-show="textmsgs" v-for="t in textmsgs" class="cslist">
                <div class="columns">
                  <span class="imtime column">{{ t.time }}</span
                  ><span class="imuser column">{{ t.user }}</span>
                </div>
                <div class="im">{{ t.message }}</div>
              </div>
            </div>
            <b-field id="send">
              <b-input
                v-model="textmsg"
                @keyup.native.enter="sendIm"
                placeholder="..."
                type="text"
              ></b-input>
              <b-button @click="sendIm" icon-left="send"> </b-button>
            </b-field>
          </div>
        </div>
      </transition>
      <transition name="fade">
        <div class="column columncenter">
          <div class="videos">
            <div
              v-show="localuser.srcObject && ssmode"
              :class="localfullscreen ? 'fullscreen' : ''"
              class="videosetforcanvas"
            >
              <div class="scale">
                <div class="v">
                  <canvas id="sscanvas" ref="sscanvas"></canvas>
                  <div class="user">
                    <div class="username">{{ localuser.userId }} (AI)</div>
                    <b-button
                      @click="localFullscreen"
                      :id="localuser.id"
                      :ref="localuser.id"
                      icon-left="fullscreen"
                      class="btnfullscreen"
                    ></b-button>
                  </div>
                </div>
              </div>
            </div>
            <div
              v-show="localuser.srcObject && !ssmode && !isPauseVideo"
              :class="localfullscreen ? 'fullscreen' : ''"
              class="videoset"
            >
              <div class="scale">
                <div class="v">
                  <video
                    ref="localvideo"
                    :src-object.prop.camel="localuser.srcObject"
                    playsinline
                    autoplay
                  ></video>
                  <div class="user">
                    <div class="username">{{ localuser.userId }}</div>
                    <b-button
                      @click="localFullscreen"
                      :id="localuser.id"
                      :ref="localuser.id"
                      icon-left="fullscreen"
                      class="btnfullscreen"
                    ></b-button>
                  </div>
                </div>
              </div>
            </div>
            <div
              v-if="users.length > 0 && u.srcObject && !u.local"
              v-for="(u, index) in users"
              :class="videofullscreen == index ? 'fullscreen' : ''"
              class="videoset"
            >
              <div class="scale">
                <div class="v">
                  <video
                    v-show="u.srcObject && !u.local"
                    :src-object.prop.camel="u.srcObject"
                    playsinline
                    autoplay
                  ></video>
                  <div class="user">
                    <div v-show="u.srcObject && !u.local" class="username">
                      {{ u.userId }}
                    </div>
                    <b-button
                      @click="videoFullscreen(index)"
                      :id="u.id"
                      :ref="u.id"
                      icon-left="fullscreen"
                      class="btnfullscreen"
                    ></b-button>
                  </div>
                </div>
              </div>
            </div>
            <div
              v-for="(u, index) in users"
              v-if="
                users.length > 0 &&
                  isScreenSharing &&
                  !isLocalScreenSharing &&
                  u.shareScreenStream
              "
              :class="videofullscreen == index ? 'fullscreen' : ''"
              class="videoset"
            >
              <div class="scale">
                <div class="v">
                  <video
                    :src-object.prop.camel="u.shareScreenStream"
                    playsinline
                    autoplay
                  ></video>
                  <div class="user">
                    <div class="username">{{ u.userId }}</div>
                    <b-button
                      @click="videoFullscreen(index)"
                      :id="u.id"
                      :ref="u.id"
                      icon-left="fullscreen"
                      class="btnfullscreen"
                    ></b-button>
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div v-if="ssmode" class="indicator">
            <div ref="fps" class="counter">
              {{ showfps }}
              <div class="title">FPS</div>
            </div>
            <div ref="inferenceTime" class="counter">
              {{ inferencetime }} <span>ms</span>
              <div class="title">Inference Time</div>
            </div>
            <div class="counter">
              <div class="value">
                <Clock />
                <span v-if="resolutionheight >= 720" class="hd">
                  <!-- <b-icon icon="high-definition" class="blicon"> </b-icon> -->
                  HD
                </span>
                <!-- <span v-if="enablevideo">
                  <b-icon icon="video" class="blicon"> </b-icon>
                  <b-icon icon="microphone" class="blicon"> </b-icon>
                </span>
                -->
                <span v-if="!enablevideo">
                  <b-icon icon="video-off" class="blicon"> </b-icon>
                  <!-- <b-icon icon="microphone" class="blicon"> </b-icon> -->
                </span>
              </div>
            </div>
          </div>
          <div v-else>
            <div class="counter less">
              <div class="value">
                <Clock />
                <span v-if="resolutionheight >= 720" class="hd">
                  <!-- <b-icon icon="high-definition" class="blicon"> </b-icon> -->
                  HD
                </span>
                <!-- <span v-if="enablevideo">
                  <b-icon icon="video" class="blicon"> </b-icon>
                  <b-icon icon="microphone" class="blicon"> </b-icon>
                </span>
                -->
                <span v-if="!enablevideo">
                  <b-icon icon="video-off" class="blicon"> </b-icon>
                  <!-- <b-icon icon="microphone" class="blicon"> </b-icon> -->
                </span>
              </div>
            </div>
          </div>
        </div>
      </transition>
      <transition name="slide-fade">
        <div
          v-if="showrightsidebar"
          class="column cr rightoptions is-one-fifth"
        >
          <div class="isleft pd cb">
            Change background
            <b-button
              id="sidebarclose"
              @click="closeRightSideBar"
              size="is-small"
              icon-left="close"
            ></b-button>
          </div>

          <div
            v-for="i in ssbgimg"
            @click="selectImg($event)"
            class="bgimgselectors"
          >
            <img :src="i" class="bgimgselector" />
          </div>

          <img id="defaultbgimg" ref="defaultbgimg" :src="defaultbgimg" />

          <div id="bgimage" class="">
            <input
              id="bgimg"
              ref="bgimg"
              @change="updateSSBackground"
              type="file"
              name="f"
              accept="image/*"
              class="inputfile inputf"
            />
            <label for="bgimg">
              <svg width="20" height="17" viewBox="0 0 20 17">
                <path
                  d="M10 0l-5.2 4.9h3.3v5.1h3.8v-5.1h3.3l-5.2-4.9zm9.3 11.5l-3.2-2.1h-2l3.4 2.6h-3.5c-.1 0-.2.1-.2.1l-.8 2.3h-6l-.8-2.2c-.1-.1-.1-.2-.2-.2h-3.6l3.4-2.6h-2l-3.2 2.1c-.4.3-.7 1-.6 1.5l.6 3.1c.1.5.7.9 1.2.9h16.3c.6 0 1.1-.4 1.3-.9l.6-3.1c.1-.5-.2-1.2-.7-1.5z"
                ></path>
              </svg>
            </label>
          </div>
        </div>
      </transition>
    </div>
    <Control ref="control" />
  </div>
</template>
<script>
import id from '~/assets/js/user/id'
export default {
  ...id
}
</script>
<style scope>
.counter {
  display: inline-block;
  font-size: 2rem;
  text-align: center;
  margin-left: 0.75rem;
  font-weight: 200;
  position: relative;
  top: -5rem;
}

.less {
  top: -2rem;
}

.blicon {
  margin-right: -4px;
}

.blicon i.mdi-24px.mdi-set,
.blicon i.mdi-24px.mdi:before {
  font-size: 18px;
}

.counter .title {
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.8rem;
  font-weight: 500;
  background-color: rgba(255, 255, 255, 0.2);
  padding: 0.2rem 1rem;
}

.counter .value {
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.8rem;
  font-weight: 500;
}

.counter span {
  font-size: 0.8rem;
  font-weight: 500;
}

.counter span.hd {
  font-size: 0.6rem;
  font-weight: bold;
  margin-left: 0.25rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 1px 6px;
  display: inline-block;
}

body {
  background: transparent;
}

.cslist {
  text-overflow: ellipsis;
  overflow: hidden;
}

.im {
  padding-right: 0.75rem;
  text-overflow: ellipsis;
  overflow: hidden;
}

.content {
  display: flex;
  flex-direction: column;
  width: 100vw;
  height: 100vh !important;
  margin-top: 0rem;
  justify-content: flex-start;
  align-items: center;
  cursor: pointer;
}

.videoset {
  display: inline-block;
  margin-bottom: -7px;
  width: calc(100% / 4.01);
  margin-right: -1px;
  overflow: hidden;
}

.isfullscreen .column .videos {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  flex-wrap: nowrap;
  overflow-y: scroll;
  overflow-x: hidden;
}

.isfullscreen .column .videoset,
.isfullscreen .column .videosetforcanvas {
  width: calc(100% / 6);
  margin-bottom: 0;
  min-height: 128px;
  opacity: 0.2;
}

.isfullscreen .column .videoset:hover,
.isfullscreen .column .videosetforcanvas:hover {
  width: calc(100% / 6);
  margin-bottom: 0;
  min-height: 128px;
  opacity: 1;
}

.videoset.fullscreen,
.videosetforcanvas.fullscreen {
  opacity: 1 !important;
}

.videosetforcanvas {
  display: inline-block;
  margin-bottom: -7px;
  width: calc(100% / 4.01);
  margin-right: -1px;
  overflow: hidden;
}

.videosetforcanvas canvas {
  width: 100%;
}

/*
#sscanvas {
  width: 100%;
  max-width: 1539px;
  max-height: 1539px;
}
*/

.fullscreen {
  width: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
}

.cl,
.cr {
  z-index: 1;
}

canvas,
video {
  z-index: 2;
}

.fullscreen canvas,
.fullscreen video {
  z-index: -1;
}

.fullscreen canvas,
.fullscreen video {
  position: fixed;
  margin: 0;
  left: 0;
  bottom: 0;
  width: 100vw !important;
  height: 100vh !important;
  object-fit: fill;
  border: 0px;
  display: block;
}

.fullscreen canvas:hover,
.fullscreen video:hover {
  transform: scale(1) !important;
}

.scale {
  width: 100%;
  padding-bottom: 56.25%;
  height: 0;
  position: relative;
}

.v {
  width: 100%;
  height: 100%;
  position: absolute;
}

.videoset .user,
.videosetforcanvas .user {
  position: relative;
  text-align: center;
  top: -7px;
  padding: 4px 0.75rem;
  margin-top: -20px;
  font-size: 0.6rem;
  height: 20px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.videoset .user:hover {
  background-color: rgba(0, 0, 0, 0.2);
}

.columncenter {
  border-left: 1px solid rgba(255, 255, 255, 0.2);
  border-right: 1px solid rgba(255, 255, 255, 0.2);
  padding: 0px;
  text-align: left;
  height: 72vh;
}

.cl {
  border-right: 0px;
}

.cr {
  border-left: 0px;
}

.upload .upload-draggable {
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 0px;
}

.rightoptions.column {
  padding: 0px;
  max-height: 72vh;
  overflow-y: scroll;
  overflow-x: hidden;
}

#bgimage {
  padding-top: 1rem;
  padding-bottom: 1rem;
}

#bgimage:hover {
  background-color: rgba(0, 0, 0, 0.2);
}

.inputfile {
  width: 0.1px;
  height: 0.1px;
  opacity: 0;
  overflow: hidden;
  position: absolute;
  z-index: -1;
}

.inputfile + label {
  max-width: 100%;
  text-overflow: ellipsis;
  white-space: nowrap;
  cursor: pointer;
  display: inline-block;
  overflow: hidden;
  padding: 0.25rem 1.25rem;
}

.inputfile:focus + label,
.inputfile.has-focus + label {
  outline: 1px dotted #000;
  outline: -webkit-focus-ring-color auto 5px;
}

.inputfile + label svg {
  width: 100%;
  height: 1.5em;
  vertical-align: middle;
  fill: currentColor;
  margin: 0;
}

.inputfile + label span {
  font-size: 0.8rem;
}

.inputf + label {
  color: rgba(255, 255, 255, 0.9);
  border: 0px solid currentColor;
}

.inputf:focus + label,
.inputf.has-focus + label,
.inputf + label:hover {
  background-color: rgba(255, 255, 255, 0);
}

.username {
  display: inline-block;
  width: 93%;
  overflow: hidden;
}

.btnfullscreen {
  color: rgba(255, 255, 255, 0.9);
  display: inline-block;
  width: 10px !important;
  height: 10px;
  margin-top: -6px;
  padding: 0px;
}

.btnfullscreen:hover {
  color: rgba(255, 255, 255, 1);
  background: transparent !important;
}

.btnfullscreen .icon:hover {
  transform: rotate(0deg) !important;
  transform: scale(1.4) !important;
}

#ssvideo {
  width: 320px;
  border: 1px solid red;
}

.cb {
  margin-bottom: 0.75rem;
}

#sidebarclose {
  float: right;
  margin-top: -3px;
}

.bgimgselectors {
  width: 50%;
  max-height: 60px;
  overflow: hidden;
  margin: 0px;
  display: inline-block;
  margin-bottom: -5px;
}

.bgimgselectors img {
  opacity: 0.8;
  height: 60px;
  margin: 0;
}

.bgimgselectors:nth-child(even) img {
  border-right: 1px solid rgba(255, 255, 255, 0.2);
}

.bgimgselectors:hover img {
  opacity: 1;
}

.bgimgselectors img {
  transition: all 0.2s ease-in-out;
}

.bgimgselectors:hover img {
  cursor: pointer;
  transform: scale(1.5);
}

.videosetforcanvas canvas,
.videoset canvas,
.videoset video {
  transition: all 0.2s ease-in-out;
  width: 100%;
  height: 100%;
  transform: scale(1);
}

.videosetforcanvas canvas:hover,
.videoset canvas:hover,
.videoset video:hover {
  cursor: pointer;
  transform: scale(1.4);
}

.bgimgselector {
  width: 100%;
}

.inputf + label {
  outline: 0;
}

#defaultbgimg {
  display: none;
}

.notices.is-bottom {
  pointer-events: inherit;
}

.notices ol {
  margin: 10px 0 0 12px;
}
</style>
