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
    private Kneejerk model;

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
    }
}
