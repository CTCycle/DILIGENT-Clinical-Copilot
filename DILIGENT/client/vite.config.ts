import path from "node:path";

import react from "@vitejs/plugin-react-swc";
import { defineConfig } from "vite";

const FASTAPI_HOST = process.env.FASTAPI_HOST || "127.0.0.1";
const FASTAPI_PORT =
  Number.parseInt(process.env.FASTAPI_PORT || "", 10) || 8000;
const FASTAPI_TARGET = `http://${FASTAPI_HOST}:${FASTAPI_PORT}`;

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/api": {
        target: FASTAPI_TARGET,
        changeOrigin: true,
        secure: false,
        rewrite: (rawPath) => rawPath.replace(/^\/api/, ""),
      },
    },
  },
  preview: {
    proxy: {
      "/api": {
        target: FASTAPI_TARGET,
        changeOrigin: true,
        secure: false,
        rewrite: (rawPath) => rawPath.replace(/^\/api/, ""),
      },
    },
  },
});
