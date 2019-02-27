const path = require('path');
const portfinder = require('portfinder');

const config = {
  entry: ['./src/WebMLPolyfill.js'],
  output: {
    filename: 'webml-polyfill.js',
    path: path.resolve(__dirname, 'dist')
  },
	module: {
		rules: [{ test: /\.js$/, loader: 'babel-loader', exclude: /node_modules/ }]
  },
  resolve: {
		extensions: ['.js']
  },
  externals: {
    'fs': true
  },
  devServer: {
    // enable https
    https: process.env.HTTPS === 'true' || false,
    // allow connections from LAN
    host: '0.0.0.0',
    // allow connections using hostname
    disableHostCheck: true,
    // serve bundle files from /dist/ without writing to disk
    publicPath: '/dist/',
  }
};

if (process.env.NODE_ENV === 'production') {
  config.mode = 'production';
  // generate a separate source map file
  // exclude it from production server if you don't want to enable source map
  config.devtool = 'source-map';
} else {
  config.mode = 'development';
  // inline the source map in bundle file for remote debugging
  config.devtool = 'inline-source-map';
}

module.exports = new Promise((resolve) => {
  const basePort = 8080;
  portfinder.getPort({
    port: basePort
  }, (_, port) => {
    config.devServer.port = port;
    config.devServer.public = `localhost:${port}`;
    resolve(config);
  });
});