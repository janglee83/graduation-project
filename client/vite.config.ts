import { defineConfig } from 'vite';
import Vue from '@vitejs/plugin-vue';
import path from 'path';

export default defineConfig({
  plugins: [
    Vue({
      template: {
        transformAssetUrls: {
          base: null,
          includeAbsolute: false,
        },
      },
    }),
  ],
  server: {
    host: true,
    port: 3000,
    open: true,
    hmr: {
      host: 'localhost',
    },
  },
  resolve: {
    extensions: ['.vue', '.ts', '.js'],
    alias: {
      '@': path.join(__dirname, '/src'),
    },
  },
});
