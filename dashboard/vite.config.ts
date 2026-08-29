import { defineConfig } from "vitest/config";
import { loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "MARKETFUSION_");
  const dashboardPort = Number(env.MARKETFUSION_DASHBOARD_PORT || "4173");
  if (!Number.isInteger(dashboardPort) || dashboardPort < 1 || dashboardPort > 65535) {
    throw new Error("MARKETFUSION_DASHBOARD_PORT must be an integer from 1 through 65535");
  }
  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: dashboardPort,
      strictPort: true,
      proxy: { "/api": "http://127.0.0.1:8765" }
    },
    preview: { host: "127.0.0.1", port: dashboardPort, strictPort: true },
    test: {
      environment: "jsdom",
      setupFiles: "./src/test/setup.ts",
      css: true,
      pool: "threads",
      maxWorkers: 1
    }
  };
});
