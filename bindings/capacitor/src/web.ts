import { WebPlugin } from '@capacitor/core'
import type { DecideOptions, DecideResult, DownloadOptions, KneejerkPlugin, LoadOptions, LoadResult } from './definitions'

/**
 * On the web there is no native library. Point `serverUrl` at a llama-server (its /v1/systemone
 * takes the same request), so the app's code path stays one; without it every call rejects.
 */
export class KneejerkWeb extends WebPlugin implements KneejerkPlugin {
  serverUrl: string | null = null
  private modelType: string | undefined

  async download(_options: DownloadOptions): Promise<{ path: string; bytes: number }> {
    throw this.unavailable('download is native only')
  }

  async load(options: LoadOptions): Promise<LoadResult> {
    if (!this.serverUrl) {
      throw this.unavailable('kneejerk runs natively on Android and iOS; on the web set serverUrl to a llama-server')
    }
    this.modelType = 'server'
    return { modelType: 'server', path: options.path }
  }

  async decide(options: DecideOptions): Promise<DecideResult> {
    if (!this.serverUrl) {
      throw this.unavailable('no native library on the web and no serverUrl set')
    }
    const t0 = performance.now()
    const r = await fetch(this.serverUrl.replace(/\/$/, '') + '/systemone', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: options.request,
    })
    return { response: await r.text(), wallMs: performance.now() - t0 }
  }

  async memory(): Promise<{ availMem: number; totalMem: number; lowMemory: boolean }> {
    return { availMem: 0, totalMem: 0, lowMemory: false }
  }

  async modelInfo(): Promise<{ info: string }> {
    throw new Error('modelInfo is only available natively')
  }

  async unload(): Promise<void> {
    this.modelType = undefined
  }

  async isLoaded(): Promise<{ loaded: boolean; modelType?: string }> {
    return { loaded: this.modelType !== undefined, modelType: this.modelType }
  }
}
