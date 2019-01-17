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
  devtool: 'source-map',
  devServer: {
    // enable https
    https: process.env.HTTPS === 'true' || false,
    // allow connections from LAN
    host: '0.0.0.0',
    // allow connections using hostname
    disableHostCheck: true,
  }
};

if (process.env.NODE_ENV === 'production') {
  config.mode = 'production';
} else {
  config.mode = 'development';
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