/** A loaded decision model: the family llama.cpp detected and the file it came from. */
export interface LoadResult {
  /** e.g. "lfm3_d1", "laya", "h2o_lightning", "clef"; "none" means the GGUF is not a decision model */
  modelType: string
  path: string
}

export interface LoadOptions {
  /** Absolute path of the GGUF file on the device (download it first, 1.1 GB for d1-2B). */
  path: string
  /** CPU threads; 0 = the library's default. On a phone 4 (the big cores) is a good start. */
  threads?: number
  /** Longest prompt in tokens (state + question + options); default 4096. */
  nCtx?: number
  /** Repack the weights for the ARM kernels: ~2.6x faster, but the model sits in RAM twice (default true). */
  repack?: boolean
  /** Absolute path of the projector GGUF (mmproj) of a multimodal model. With it, a request can carry "images": ["data:image/jpeg;base64,..."]. */
  mmproj?: string
}

export interface DecideOptions {
  /**
   * The request body of llama-server's /v1/systemone, as a JSON string:
   * {"state": "...", "questions": {"id": {"type": "noul"|"choice"|"score", "instructions": "...", "criteria": ...}}}
   */
  request: string
}

export interface DecideResult {
  /** The /v1/systemone answer as a JSON string: {"model", "answers", "usage"} or {"error": {...}}. */
  response: string
  /** Wall time of the decision in milliseconds. */
  wallMs: number
}

export interface KneejerkPlugin {
  /** Loads a GGUF decision model. One model at a time; loading another unloads the first. */
  load(options: LoadOptions): Promise<LoadResult>
  /** Answers one request. Rejects when no model is loaded. */
  decide(options: DecideOptions): Promise<DecideResult>
  /** Frees the model and its memory. */
  /** Free and total RAM in bytes; a model with repacking takes about twice its file size. */
  memory(): Promise<{ availMem: number; totalMem: number; lowMemory: boolean }>
  /** What the model is made of: architecture, sizes, one entry per layer (JSON text, see systemone_model_info). */
  modelInfo(): Promise<{ info: string }>
  unload(): Promise<void>
  isLoaded(): Promise<{ loaded: boolean; modelType?: string }>
}
