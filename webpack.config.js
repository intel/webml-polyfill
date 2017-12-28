const path = require('path');

module.exports = {
  entry: './src/WebMLPolyfill.js',
  output: {
    filename: 'webml-polyfill.js',
    path: path.resolve(__dirname, 'dist')
  },
	module: {
		rules: [
			{
			test: /\.js$/,
			include: [
				path.resolve(__dirname, "src"),
			],
			use: {
				loader: 'babel-loader',
				options: {
				presets: ['env']
				}
			}
			}
		]
  },
  resolve: {
	extensions: ['.js']
  }  
};
