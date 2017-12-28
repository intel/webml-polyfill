const path = require('path');

module.exports = {
  entry: './src/WebMLPolyfill.js',
  output: {
    filename: 'webml-polyfill.js',
    path: path.resolve(__dirname, 'dist')
  },
	module: {
		rules: [{ test: /\.js$/, loader: 'babel-loader', exclude: /node_modules/ }]
  },
  resolve: {
		extensions: ['.js']
  }  
};
