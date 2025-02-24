import { defineConfig } from 'vite'
import deno from '@deno/vite-plugin'
import solid from 'vite-plugin-solid'


export default defineConfig({
    plugins: [deno(), solid()],
    server: {
        mimeTypes: {
            'application/wasm': ['wasm'],
        },
    },
    build: {
        target: "esnext"
    }
})
