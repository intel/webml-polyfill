/* eslint-disable prettier/prettier */
const fs = require('fs')
const config = require('./config.js')
/* eslint-disable prettier/prettier */
module.exports = {
  mode: 'universal',
  /*
   ** Headers of the page
   */
  head: {
    title: 'WebNN Meeting - meet you, meet happily',
    meta: [
      { charset: 'utf-8' },
      { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      {
        hid: 'description',
        name: 'description',
        content: process.env.npm_package_description || ''
      }
    ],
    link: [
      { rel: 'apple-touch-icon', type: 'image/png', size: '180x180', href: '/apple-touch-icon.png' },
      { rel: 'icon', type: 'image/png', size: '32x32', href: '/favicon-32x32.png' },
      { rel: 'icon', type: 'image/png', size: '16x16', href: '/favicon-16x16.png' },
      { rel: 'manifest', href: '/site.webmanifest' },
      {
        rel: 'stylesheet',
        type: 'text/css',
        href:
          '../../css/font/font.css?family=Lato:300,300i,400,400i&display=swap'
      }
    ]
  },
  /*
   ** Customize the progress-bar color
   */
  loading: { color: '#fff' },
  router: {
    middleware: ['dynamiclayout']
  },
  /*
   ** Global CSS
   */
  css: [
    // 直接加载一个 Node.js 模块。（在这里它是一个 Sass 文件）
    'bulma',
    // 项目里要用的 CSS 文件
    '@/assets/css/main.css',
    // 项目里要使用的 SCSS 文件
    '@/assets/css/main.scss'
  ],
  /*
   ** Plugins to load before mounting the App
   */
  // plugins: [{ src: '~/plugins/vue-js-grid.js', mode: 'client' }],
  /*
   ** Nuxt.js dev-modules
   */
  buildModules: [
    // Doc: https://github.com/nuxt-community/eslint-module
    '@nuxtjs/eslint-module',
    // Doc: https://github.com/nuxt-community/stylelint-module
    '@nuxtjs/stylelint-module'
  ],
  /*
   ** Nuxt.js modules
   */
  modules: [
    // Doc: https://buefy.github.io/#/documentation
    ['nuxt-buefy', { css: false, materialDesignIcons: true }],
    // Doc: https://axios.nuxtjs.org/usage
    '@nuxtjs/axios',
    '@nuxtjs/pwa',
    // Doc: https://github.com/nuxt-community/dotenv-module
    // '@nuxtjs/dotenv',
    // '~/modules/authrequest'
    // // Passing options
    // '~/modules/authrequest', { token: '123' }
  ],
  /*
   ** Axios module configuration
   ** See https://axios.nuxtjs.org/options
   */
  axios: {},
  /*
   ** Build configuration
   */
  build: {
    /*
     ** You can extend webpack config here
     */
    babel: {
      sourceType: 'unambiguous'
    },
    extend(config, ctx) {}
  },
  server: {
    port: config.nuxtserver.httpsport, // default: 3000
    host: config.nuxtserver.host, // default: localhost,
    timing: false,
    https: {
      key: fs.readFileSync(config.certificate.key),
      cert: fs.readFileSync(config.certificate.cert)
    }
  }
}
