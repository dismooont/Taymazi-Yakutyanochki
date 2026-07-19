import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// В разработке фронт и API живут на разных портах, но cookie сессии — HttpOnly
// и SameSite=Lax, поэтому запросы должны выглядеть одноисточниковыми. Прокси
// решает это без ослабления настроек cookie.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
