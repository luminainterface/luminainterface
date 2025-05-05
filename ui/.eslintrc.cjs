module.exports = {
  root: true,
  env: {
    node: true,
    'vue/setup-compiler-macros': true
  },
  extends: [
    'plugin:vue/vue3-essential',
    'eslint:recommended',
    '@vue/typescript/recommended',
    '@vue/prettier',
    '@vue/prettier/@typescript-eslint'
  ],
  parserOptions: {
    ecmaVersion: 2020
  },
  rules: {
    'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
    'no-debugger': process.env.NODE_ENV === 'production' ? 'warn' : 'off'
  },
  overrides: [
    {
      files: ['cypress/**/*.ts'],
      env: { 
        mocha: true,
        'cypress/globals': true
      },
      rules: { 
        'no-unused-expressions': 'off',
        '@typescript-eslint/no-namespace': 'off'
      },
      extends: [
        'plugin:cypress/recommended'
      ]
    }
  ]
} 