<template>
  <div class="hero-head">
    <section class="section container">
      <b-navbar :class="getFullscreen ? 'fullscreen' : ''">
        <template slot="brand">
          <b-navbar-item :to="{ path: '/' }" class="logo" tag="router-link">
            <div class="titlea">
              WebNN Meeting
              <div class="slogan lg">
                meet you, meet happily
              </div>
            </div>
          </b-navbar-item>
        </template>
        <template slot="end">
          <!-- <b-navbar-dropdown label="Theme">
            <b-navbar-item id="scenery" @click="setLayout" href="#">
              Scenery
            </b-navbar-item>
            <b-navbar-item id="dynamic" @click="setLayout" href="#">
              Dynamic
            </b-navbar-item>
            <b-navbar-item id="immersive" href="#">
              Immersive
            </b-navbar-item>
          </b-navbar-dropdown>
          <b-navbar-dropdown label="EN">
            <b-navbar-item href="#">
              English
            </b-navbar-item>
            <b-navbar-item href="#">
              Simplified Chinese
            </b-navbar-item>
          </b-navbar-dropdown>
          -->
          <div class="backendlogo">
            <div v-if="backend === 'wasm'" class="backendwasm">
              <svg id="wasm">
                <path
                  d="m376 0c0 1.08 0 2.16 0 3.3 0 38.76-31.42 70.17-70.17 70.17-38.76 0-70.17-31.42-70.17-70.17l0 0c0-1.14 0-2.22 0-3.3L0 0l0 612 612 0 0-612z"
                  fill="rgba(255, 255, 255, 0.2)"
                />
                <path
                  d="m142.16 329.81 40.56 0 27.69 147.47 0.5 0 33.28-147.47 37.94 0 30.06 149.28 0.59 0 31.56-149.28 39.78 0-51.69 216.69-40.25 0-29.81-147.47-0.78 0-31.91 147.47-41 0zm287.69 0 63.94 0 63.5 216.69-41.84 0-13.81-48.22-72.84 0-10.66 48.22-40.75 0zm24.34 53.41-17.69 79.5 55.06 0-20.31-79.5z"
                  fill="rgba(255, 255, 255, 0.8)"
                />
              </svg>
            </div>
            <div v-if="backend === 'webgl'" class="backendwebgl">
              <svg id="webgl" viewBox="0 0 1442.9733 602.98669">
                <g
                  id="g10"
                  transform="matrix(1.3333333,0,0,-1.3333333,0,602.98667)"
                >
                  <g id="g12" transform="scale(0.10)">
                    <path
                      id="path14"
                      style="fill: rgba(255, 255, 255, 0.8); fill-opacity: 1; fill-rule: nonzero; stroke: none"
                      d="m 10519,1519.96 h -73.4 v -191.11 h -64.2 v 191.11 h -73.5 v 55.21 h 211.1 v -55.21 0"
                    />
                    <path
                      id="path16"
                      style="fill: rgba(255, 255, 255, 0.8); fill-opacity: 1; fill-rule: nonzero; stroke: none"
                      d="m 10547.7,1575.17 h 94.5 l 42.4,-164.21 h 0.7 l 42.5,164.21 h 94.5 v -246.32 h -60.4 v 186.97 h -0.7 l -51.7,-186.97 h -49 l -51.8,186.97 h -0.7 v -186.97 h -60.3 v 246.32"
                    />
                    <path
                      id="path18"
                      style="fill: rgba(255, 255, 255, 0.8); fill-opacity: 1; fill-rule: nonzero; stroke: none"
                      d="m 7978.08,1342.75 c -89.56,-37.77 -180,-56.68 -271.24,-56.68 -144.07,0 -273.56,25.11 -388.6,75.22 -115.05,50.14 -212.06,119.18 -291.08,207.22 -79.06,87.94 -139.61,191.33 -181.75,310.1 -42.15,118.77 -63.24,246.84 -63.24,384.09 0,140.7 21.02,271.37 63.05,391.93 42.01,120.51 102.46,225.63 181.31,315.39 78.85,89.75 175.73,160.11 290.71,211.19 114.99,51.01 244.47,76.56 388.54,76.56 96.57,0 190.04,-14.57 280.49,-43.64 90.44,-29.14 172.07,-71.91 244.97,-128.31 72.89,-56.42 133.03,-125.97 180.44,-208.59 47.43,-82.69 76.43,-177.54 86.98,-284.64 h -395.1 c -24.61,105.31 -72.04,184.34 -142.27,237.03 -70.31,52.69 -155.47,79.04 -255.51,79.04 -93.11,0 -172.13,-18.04 -237.1,-54.12 -64.95,-36.07 -117.65,-84.49 -158.06,-145.17 -40.42,-60.75 -69.8,-129.8 -88.21,-207.23 -18.48,-77.42 -27.64,-157.44 -27.64,-240.13 0,-79.22 9.22,-156.21 27.69,-230.96 18.49,-74.77 47.92,-142.09 88.34,-201.89 40.41,-59.82 93.17,-107.73 158.25,-143.81 65.02,-36.08 144.17,-54.12 237.42,-54.12 137.12,0 242.98,34.59 317.73,103.76 74.77,69.19 118.26,169.48 130.62,300.88 h -416.19 v 309.45 h 789.09 V 1328.3 h -262.57 l -42.03,213.36 c -73.76,-94.84 -155.47,-161.17 -245.04,-198.91"
                    />
                    <path
                      id="path20"
                      style="fill: rgba(255, 255, 255, 0.8); fill-opacity: 1; fill-rule: nonzero; stroke: none"
                      d="M 9198.36,3210.16 V 1678.83 h 916.74 V 1329.72 H 8785.78 v 1880.44 h 412.58"
                    />
                    <path
                      id="path22"
                      style="fill: rgba(255, 255, 255, 0.8); fill-opacity: 1; fill-rule: nonzero; stroke: none"
                      d="M 7247.14,1095.97 C 6615.86,712.02 5670.38,493.484 4615.45,493.484 c -1900.61,0 -3441.37,791.426 -3441.37,1767.726 0,976.28 1540.76,1767.72 3441.37,1767.72 1058.79,0 2008.13,-222.6 2639.42,-608.95 C 6565.8,4067.31 5330.32,4522.18 3917.71,4522.39 1754.21,4522.75 0.1875,3510.64 0,2261.82 -0.183594,1013.04 1753.52,0.320313 3917.02,0 c 1407,-0.226563 2639.27,453.164 3330.12,1095.97"
                    />
                    <path
                      id="path24"
                      style="fill: rgba(255, 255, 255, 0.8); fill-opacity: 1; fill-rule: nonzero; stroke: none"
                      d="M 3779.98,3356.47 3417.18,1786.54 2979.1,3356.47 H 2646.44 L 2209.34,1786.52 1845.46,3356.47 H 1502.7 l 538.32,-2046.3 h 321.59 l 452.86,1607.25 447.4,-1607.25 h 322.52 l 557.24,2046.3 h -362.65"
                    />
                    <path
                      id="path26"
                      style="fill: rgba(255, 255, 255, 0.8); fill-opacity: 1; fill-rule: nonzero; stroke: none"
                      d="m 5049.1,2495.18 c -50.38,70.9 -113.39,125.47 -187.33,162.22 -76.03,37.8 -164.18,56.95 -261.98,56.95 -110.08,0 -209.26,-19.96 -294.78,-59.33 -81.1,-37.35 -150.33,-92.25 -205.77,-163.2 -102.61,-131.3 -156.85,-316.31 -156.85,-534.98 0,-97.33 13.81,-188.65 41.04,-271.45 27.39,-83.24 67.82,-156.31 120.17,-217.14 110.07,-127.9 266.99,-195.51 453.8,-195.51 98.09,0 178.18,10.26 244.8,31.36 58.23,18.42 96.92,42.29 128.61,63.22 40.84,26.94 78.35,60.06 111.53,98.39 28.16,32.55 53.24,68.91 74.56,108.05 38.17,70.11 56.65,135.6 60.85,170.38 l 2.79,23 H 4913.3 l -1.47,-18.94 c -6.74,-86.82 -120.7,-241.93 -323.73,-241.93 -290.24,0 -359.39,209.82 -363.98,394.58 h 971.69 v 20.53 c 0,119.15 -12.19,227.94 -36.24,323.37 -24.59,97.62 -61.75,181.88 -110.47,250.43 z m -480.01,-21.95 c 215.27,0 330.36,-115.87 351.32,-354.05 h -696 c 10.2,193.95 163.89,354.05 344.68,354.05"
                    />
                    <path
                      id="path28"
                      style="fill: rgba(255, 255, 255, 0.8); fill-opacity: 1; fill-rule: nonzero; stroke: none"
                      d="m 6540.78,2298.41 c -25.88,82.68 -64.03,155.53 -113.34,216.55 -50.93,62.99 -112.7,111.98 -183.58,145.61 -74,35.11 -156.95,52.91 -246.53,52.91 H 5997 c -82.65,-1.34 -158.12,-19.36 -224.27,-53.58 -54.7,-28.27 -103.72,-67.96 -146.22,-118.32 v 627.78 H 5360.37 V 1309.85 h 253.36 v 126.56 c 22.24,-29.1 50.82,-59.8 88.68,-86.93 71.86,-51.5 160.76,-77.6 264.23,-77.6 111.29,0 210.5,24.82 294.92,73.8 75.24,43.64 138.94,106.52 189.35,186.86 44.01,70.14 77.5,152.75 99.5,245.55 19.01,80.22 29.07,168 29.07,253.84 0,94.85 -13.02,184.5 -38.7,266.48 z m -239.14,-301.94 c 0,-145.99 -29.36,-268.34 -84.94,-353.83 -59.17,-91.08 -145.04,-137.24 -255.18,-137.24 -159.26,0 -345.23,111.37 -345.23,425.23 0,150.33 18.58,265.99 56.79,353.65 55.18,126.56 147.87,188.08 283.33,188.08 144.15,0 240.56,-56.5 294.73,-172.72 44.75,-96.05 50.5,-212.47 50.5,-303.17"
                    />
                  </g>
                </g>
              </svg>
            </div>
          </div>
        </template>
      </b-navbar>
    </section>
  </div>
</template>
<script>
export default {
  data() {
    return {
      backend: null
    }
  },
  computed: {
    getFullscreen() {
      return this.$store.state.fullscreen
    }
  },
  mounted() {
    this.checkBackend()
  },
  methods: {
    setLayout(e) {
      this.$store.commit('setLayout', e.target.id)
      this.$router.go(0)
    },
    checkBackend() {
      if (this.$route.query.b) {
        this.backend = this.$route.query.b.toLowerCase().trim()
      }
    }
  }
}
</script>
<style scope>
.backendlogo {
  display: inline-block;
  width: 55px;
  height: 55px;
}

.backendwasm {
  display: inline-block;
  width: 55px;
  height: 55px;
  padding: 16px 0px 0 10px;
}

.backendwebgl {
  height: 55px;
  width: 55px;
  padding: 24px 0 0 0;
  margin-left: -16px;
}

.backendwebgl svg#webgl {
  transform: scale(1.2);
}

.backendwasm svg#wasm {
  display: inline-block;
}

.backendwasm #wasm path {
  transform: scale(0.06);
}

nav.fullscreen {
  display: none;
}

.navbar-item,
.navbar-link {
  color: rgba(255, 255, 255, 1);
}

.titlea {
  font-size: 1.5rem;
  font-style: italic;
  display: block;
  width: 100%;
}

.slogan {
  display: block;
  font-weight: 400;
  font-style: italic;
  font-size: 0.8rem;
  letter-spacing: -0.01em;
  width: 100%;
}

@media screen and (min-width: 1024px) {
  .navbar-dropdown {
    padding: 0;
  }

  .navbar-item:focus,
  .navbar-item:hover,
  .navbar-item.is-active {
    background-color: rgba(0, 0, 0, 0.1);
  }

  .navbar-item.has-dropdown:focus .navbar-link,
  .navbar-item.has-dropdown:hover .navbar-link,
  .navbar-item.has-dropdown.is-active .navbar-link {
    background-color: rgba(0, 0, 0, 0.1);
  }

  .navbar-dropdown a.navbar-item {
    padding: 0.375rem 1rem;
    background-color: rgba(0, 0, 0, 0.1);
  }

  .navbar-dropdown a.navbar-item:hover {
    padding: 0.375rem 1rem;
    background-color: rgba(0, 0, 0, 0.2);
  }

  .navbar-dropdown {
    background-color: transparent;
    border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
    border-top: 1px solid rgba(204, 255, 144, 1);
    box-shadow: 0 8px 8px rgba(10, 10, 10, 0.1);
    display: none;
    font-size: 0.875rem;
    left: 0;
    min-width: 100%;
    position: absolute;
    top: 100%;
    z-index: 20;
  }
}

@media (max-width: 768px) {
  .navbar-menu {
    background-color: rgba(0, 0, 0, 0.2);
  }

  .logo {
    background-color: transparent;
  }

  a.navbar-item:focus,
  a.navbar-item:focus-within,
  a.navbar-item:hover,
  a.navbar-item.is-active,
  .navbar-link:focus,
  .navbar-link:focus-within,
  .navbar-link:hover,
  .navbar-link.is-active {
    background-color: rgba(0, 0, 0, 0.4);
  }

  .navbar-dropdown {
    padding: 0;
  }
}
</style>
