import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // Split heavy, rarely-changing vendor code into its own chunks
        // so the browser can cache them independently of app code, and
        // so the initial page load doesn't need to fetch charting code
        // it isn't using yet (the projects list page never renders a chart).
        manualChunks(id: string) {
          if (id.includes('recharts') || id.includes('d3-')) return 'vendor-charts';
          if (id.includes('node_modules')) return 'vendor';
        },
      },
    },
  },
})
