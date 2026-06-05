import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  integrations: [tailwind()],
  base: '/',
  build: {
    format: 'file',
  },
  trailingSlash: 'never',
  vite: {
    build: {
      rollupOptions: {
        output: {
          assetFileNames: '_astro/styles[extname]',
        },
      },
    },
  },
});
