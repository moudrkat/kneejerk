package com.moudrkat.kneejerk;

import android.os.SystemClock;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Capacitor side: load / decide / unload / isLoaded, one model at a time. Native work runs on one
 * background thread so a 1 GB load or a decision never blocks the WebView; calls are answered in order.
 */
@CapacitorPlugin(name = "Kneejerk")
public class KneejerkPlugin extends Plugin {
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final ExecutorService downloader = Executors.newSingleThreadExecutor();
    private Kneejerk model;

    /**
     * Downloads a model file so that an interrupted download continues where it stopped: bytes go to
     * path + ".part", every new attempt asks the server for the rest (HTTP Range), and the file gets its
     * name only when complete. A dropped connection is retried here; after the app was killed, calling
     * download again picks the partial file up.
     */
    @PluginMethod
    public void download(PluginCall call) {
        final String url = call.getString("url");
        final String path = call.getString("path");
        final int retries = call.getInt("retries", 20);
        if (url == null || url.isEmpty() || path == null || path.isEmpty()) {
            call.reject("url and path are required");
            return;
        }
        downloader.execute(() -> {
            try {
                final long[] lastEvent = {0};
                final long bytes = ResumableDownload.fetch(url, path, retries, (have, total) -> {
                    final long now = SystemClock.elapsedRealtime();
                    if (now - lastEvent[0] > 300) {
                        lastEvent[0] = now;
                        JSObject progress = new JSObject();
                        progress.put("path", path);
                        progress.put("bytes", have);
                        progress.put("total", total);
                        notifyListeners("downloadProgress", progress);
                    }
                });
                JSObject ret = new JSObject();
                ret.put("path", path);
                ret.put("bytes", bytes);
                call.resolve(ret);
            } catch (Exception e) {
                call.reject("download failed: " + e.getMessage());
            }
        });
    }

    @PluginMethod
    public void load(PluginCall call) {
        final String path = call.getString("path");
        if (path == null || path.isEmpty()) {
            call.reject("path is required");
            return;
        }
        final int threads = call.getInt("threads", 0);
        final int nCtx = call.getInt("nCtx", 4096);
        final boolean repack = call.getBoolean("repack", true);
        final String mmproj = call.getString("mmproj");
        worker.execute(() -> {
            try {
                if (model != null) {
                    model.close();
                    model = null;
                }
                model = Kneejerk.load(path, threads, nCtx, repack, mmproj);
                JSObject ret = new JSObject();
                ret.put("modelType", model.modelType());
                ret.put("path", path);
                call.resolve(ret);
            } catch (Throwable t) {
                call.reject(t.getMessage() != null ? t.getMessage() : t.toString());
            }
        });
    }

    @PluginMethod
    public void decide(PluginCall call) {
        final String request = call.getString("request");
        if (request == null) {
            call.reject("request (a /v1/systemone JSON body) is required");
            return;
        }
        worker.execute(() -> {
            if (model == null) {
                call.reject("no model loaded");
                return;
            }
            try {
                long t0 = SystemClock.elapsedRealtimeNanos();
                String response = model.decide(request);
                double ms = (SystemClock.elapsedRealtimeNanos() - t0) / 1e6;
                JSObject ret = new JSObject();
                ret.put("response", response);
                ret.put("wallMs", ms);
                call.resolve(ret);
            } catch (Throwable t) {
                call.reject(t.getMessage() != null ? t.getMessage() : t.toString());
            }
        });
    }

    /** Free and total RAM in bytes, so an app can decide whether a model fits (with repacking it takes about twice its file). */
    @PluginMethod
    public void memory(PluginCall call) {
        android.app.ActivityManager am = (android.app.ActivityManager) getContext().getSystemService(android.content.Context.ACTIVITY_SERVICE);
        android.app.ActivityManager.MemoryInfo mi = new android.app.ActivityManager.MemoryInfo();
        am.getMemoryInfo(mi);
        JSObject ret = new JSObject();
        ret.put("availMem", mi.availMem);
        ret.put("totalMem", mi.totalMem);
        ret.put("lowMemory", mi.lowMemory);
        call.resolve(ret);
    }

    @PluginMethod
    public void modelInfo(PluginCall call) {
        worker.execute(() -> {
            if (model == null) {
                call.reject("no model loaded");
                return;
            }
            try {
                JSObject ret = new JSObject();
                ret.put("info", model.modelInfo());
                call.resolve(ret);
            } catch (Throwable t) {
                call.reject(t.getMessage() != null ? t.getMessage() : t.toString());
            }
        });
    }

    @PluginMethod
    public void unload(PluginCall call) {
        worker.execute(() -> {
            if (model != null) {
                model.close();
                model = null;
            }
            call.resolve();
        });
    }

    @PluginMethod
    public void isLoaded(PluginCall call) {
        JSObject ret = new JSObject();
        ret.put("loaded", model != null);
        if (model != null) {
            ret.put("modelType", model.modelType());
        }
        call.resolve(ret);
    }

    @Override
    protected void handleOnDestroy() {
        worker.execute(() -> {
            if (model != null) {
                model.close();
                model = null;
            }
        });
        worker.shutdown();
        downloader.shutdownNow();
    }
}
