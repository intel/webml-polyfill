const path = require('path');
const webpack = require('webpack')

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
  }
};

if (process.env.NODE_ENV === 'production') {
  config.plugins = [
    new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('production') }),
    new webpack.optimize.ModuleConcatenationPlugin(),
    new webpack.optimize.UglifyJsPlugin({
      compress: { warnings: false, unused: false },
      output: { comments: false }
    })
  ]
} else {
  config.plugins = [new webpack.DefinePlugin({ 'process.env.NODE_ENV': JSON.stringify('development') })]
}

module.exports = config