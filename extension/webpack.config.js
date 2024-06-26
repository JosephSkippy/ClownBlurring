const path = require('path');

module.exports = {
  entry: {
    background: './src/background.js',
    content: './src/content.js',
  },
  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env'],
          },
        },
      },
    ],
  },
  resolve: {
    extensions: ['.js', '.json', '.wasm'],
  },
  mode: 'development', // Change to 'production' for production builds
  devtool: 'source-map', // Use 'source-map' instead of 'eval'
};