module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      '@babel/plugin-transform-private-methods',
      'react-native-reanimated/plugin',
    ],
  };
};
