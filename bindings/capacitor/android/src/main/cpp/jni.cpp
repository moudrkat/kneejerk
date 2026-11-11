// JNI bridge: the C functions of systemone.h as static natives of com.moudrkat.kneejerk.Kneejerk.
// Strings cross as Java Strings (UTF-8 both ways); the model handle crosses as a long.

#include <jni.h>
#include <android/log.h>

#include <cstring>
#include <string>

#include "systemone.h"

#define LOG_TAG "kneejerk"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::string from_java(JNIEnv * env, jstring s) {
    if (!s) {
        return {};
    }
    const char * utf = env->GetStringUTFChars(s, nullptr);
    std::string out = utf ? utf : "";
    env->ReleaseStringUTFChars(s, utf);
    return out;
}

// NewStringUTF chokes on supplementary characters in Java's modified UTF-8; build the String from bytes instead.
jstring to_java(JNIEnv * env, const std::string & s) {
    jbyteArray bytes = env->NewByteArray((jsize) s.size());
    env->SetByteArrayRegion(bytes, 0, (jsize) s.size(), reinterpret_cast<const jbyte *>(s.data()));
    jclass string_class = env->FindClass("java/lang/String");
    jmethodID ctor = env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
    jstring charset = env->NewStringUTF("UTF-8");
    jstring out = (jstring) env->NewObject(string_class, ctor, bytes, charset);
    env->DeleteLocalRef(bytes);
    env->DeleteLocalRef(charset);
    env->DeleteLocalRef(string_class);
    return out;
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_moudrkat_kneejerk_Kneejerk_nativeLoad(JNIEnv * env, jclass, jstring path, jint threads, jint n_ctx, jboolean repack, jstring mmproj) {
    systemone_params p = systemone_default_params();
    p.n_gpu_layers = 0;              // CPU on Android; GPU backends are a later experiment
    p.n_threads    = threads;
    p.n_ctx        = n_ctx > 0 ? (uint32_t) n_ctx : 4096;
    p.repack       = repack == JNI_TRUE;
    p.flash_attn   = false;          // the attention maps are part of what the app shows
    const std::string file = from_java(env, path);
    const std::string projector = from_java(env, mmproj);
    p.mmproj_path = projector.empty() ? nullptr : projector.c_str(); // with it, requests can carry images
    LOGI("loading %s (threads %d, n_ctx %u, repack %d)", file.c_str(), p.n_threads, p.n_ctx, (int) p.repack);
    systemone * so = systemone_load(file.c_str(), &p);
    if (!so) {
        LOGE("load failed: %s", file.c_str());
        return 0;
    }
    LOGI("loaded, decision model type %s", systemone_model_type(so));
    return reinterpret_cast<jlong>(so);
}

JNIEXPORT jstring JNICALL
Java_com_moudrkat_kneejerk_Kneejerk_nativeModelType(JNIEnv * env, jclass, jlong handle) {
    auto * so = reinterpret_cast<systemone *>(handle);
    return to_java(env, so ? systemone_model_type(so) : "none");
}

JNIEXPORT jstring JNICALL
Java_com_moudrkat_kneejerk_Kneejerk_nativeDecide(JNIEnv * env, jclass, jlong handle, jstring request) {
    auto * so = reinterpret_cast<systemone *>(handle);
    const std::string req = from_java(env, request);
    char * answer = systemone_decide(so, req.c_str());
    std::string out = answer ? answer : "{\"error\":{\"message\":\"no answer\",\"type\":\"server_error\"}}";
    systemone_free_string(answer);
    return to_java(env, out);
}

JNIEXPORT jstring JNICALL
Java_com_moudrkat_kneejerk_Kneejerk_nativeModelInfo(JNIEnv * env, jclass, jlong handle) {
    auto * so = reinterpret_cast<systemone *>(handle);
    char * info = systemone_model_info(so);
    std::string out = info ? info : "{}";
    systemone_free_string(info);
    return to_java(env, out);
}

JNIEXPORT void JNICALL
Java_com_moudrkat_kneejerk_Kneejerk_nativeFree(JNIEnv *, jclass, jlong handle) {
    systemone_free(reinterpret_cast<systemone *>(handle));
}

} // extern "C"
