package com.moudrkat.kneejerk;

/** The native library: one loaded decision model, answering /v1/systemone requests. Not thread-safe; the plugin serialises calls. */
public final class Kneejerk implements AutoCloseable {
    static {
        System.loadLibrary("kneejerk_jni");
    }

    private long handle;
    private final String modelType;
    private final String path;

    private Kneejerk(long handle, String path) {
        this.handle = handle;
        this.path = path;
        this.modelType = nativeModelType(handle);
    }

    /** Loads a GGUF; throws when the file cannot be loaded or is not a decision model llama.cpp knows. */
    public static Kneejerk load(String path, int threads, int nCtx, boolean repack) {
        return load(path, threads, nCtx, repack, null);
    }

    /** With mmproj (the projector GGUF of a multimodal model), requests can carry "images": [data URLs]. */
    public static Kneejerk load(String path, int threads, int nCtx, boolean repack, String mmproj) {
        long h = nativeLoad(path, threads, nCtx, repack, mmproj == null ? "" : mmproj);
        if (h == 0) {
            throw new IllegalStateException("kneejerk: failed to load " + path);
        }
        Kneejerk k = new Kneejerk(h, path);
        if ("none".equals(k.modelType)) {
            k.close();
            throw new IllegalArgumentException("kneejerk: not a decision model: " + path);
        }
        return k;
    }

    public String modelType() { return modelType; }
    public String path() { return path; }

    /** The /v1/systemone request as JSON text; returns the answer (or an {"error": ...}) as JSON text. */
    public synchronized String decide(String requestJson) {
        if (handle == 0) {
            throw new IllegalStateException("kneejerk: model is unloaded");
        }
        return nativeDecide(handle, requestJson);
    }

    /** What the model is made of (architecture, sizes, one entry per layer), as JSON text. */
    public synchronized String modelInfo() {
        if (handle == 0) {
            throw new IllegalStateException("kneejerk: model is unloaded");
        }
        return nativeModelInfo(handle);
    }

    @Override
    public synchronized void close() {
        if (handle != 0) {
            nativeFree(handle);
            handle = 0;
        }
    }

    private static native long nativeLoad(String path, int threads, int nCtx, boolean repack, String mmproj);
    private static native String nativeModelType(long handle);
    private static native String nativeDecide(long handle, String requestJson);
    private static native String nativeModelInfo(long handle);
    private static native void nativeFree(long handle);
}
