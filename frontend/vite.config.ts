import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The production bundle is written into the Python package so `voicerail web`
// serves it without Node being installed.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  build: {
    outDir: "../src/voicerail/web/static",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8080" },
  },
});
