import * as path from 'node:path'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

const defaultHost = '127.0.0.1'
const defaultUiPort = 7861
const defaultApiPort = 8000

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, path.resolve(__dirname, '../settings'), '')
    const apiHost = env.FASTAPI_HOST || defaultHost
    const apiPort = Number.parseInt(env.FASTAPI_PORT || `${defaultApiPort}`, 10) || defaultApiPort
    const uiHost = env.UI_HOST || defaultHost
    const uiPort = Number.parseInt(env.UI_PORT || `${defaultUiPort}`, 10) || defaultUiPort
    const apiTarget = `http://${apiHost}:${apiPort}`

    return {
        envDir: path.resolve(__dirname, '../settings'),
        plugins: [react()],
        server: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: {
                '/api': {
                    target: apiTarget,
                    changeOrigin: true,
                    rewrite: (proxyPath) => proxyPath.replace(/^\/api/, ''),
                },
            },
        },
        preview: {
            host: uiHost,
            port: uiPort,
            strictPort: false,
            proxy: {
                '/api': {
                    target: apiTarget,
                    changeOrigin: true,
                    rewrite: (proxyPath) => proxyPath.replace(/^\/api/, ''),
                },
            },
        },
    }
})
