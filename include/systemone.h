// libsystemone: decision models (TypeSafe /v1/systemone) as a library, on llama.cpp.
//
// One request is the JSON body llama-server's /v1/systemone takes: a "state" and typed "questions"
// (noul, choice, score). The answer is the JSON llama-server returns: calibrated probabilities per
// question, no token generated. Text only: images and audio are left to the server for now.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct systemone systemone;

typedef struct systemone_params {
    int32_t n_gpu_layers;   // layers on the GPU, 0 = CPU only, -1 = all
    uint32_t n_ctx;         // longest prompt in tokens (state + question + options); 0 = 4096
    int32_t n_threads;      // CPU threads; 0 = hardware default
    bool verbose;           // llama.cpp logs to stderr
    bool repack;            // CPU weight repacking: faster kernels, but the weights are held twice in RAM (default true)
    bool flash_attn;        // fused attention (default true); false materialises the attention maps, which "layers" then returns
    const char * mmproj_path; // the projector (mmproj GGUF) of a multimodal model, NULL = text only. With it, a request
                              // can carry "images": ["data:image/jpeg;base64,..."] (d1, d1-omni, ...)
} systemone_params;

systemone_params systemone_default_params(void);

// Loads a GGUF decision model. NULL on failure, the reason is printed to stderr (there is no handle
// to ask systemone_last_error() about).
systemone * systemone_load(const char * model_path, const systemone_params * params);
void systemone_free(systemone * so);

// The decision model family llama.cpp detected, e.g. "lfm3_d1", "openjev"; "none" for a model that is not one.
const char * systemone_model_type(const systemone * so);

// Answers one request. Returns a malloc'd JSON string: {"model", "answers", "usage"} as the server, or
// an extension of this library: "layers": true in the request adds to every answer the answer each layer
// would give (the residual stream after the layer through the output head, a logit lens): {"options",
// "probabilities": [[per option] per layer], "ms": [when each layer was done]} or [noul per layer].
// Label-token models only (d1, ...). Prompts up to 512 tokens also get "tokens" (the prompt as the model
// reads it, one piece per token), "token_change": [[per token] per layer] (how much each layer moved each
// token, 0..1 within the layer) and, with flash_attn off, "attention": {"<layer>": [[per token] per head]}
// (where the last position looked, every head).
// Returns a malloc'd JSON string: {"model", "answers", "usage"} as the server, or
// {"error": {"message", "type"}}. Free it with systemone_free_string(). NULL only if out of memory.
// Safe to call from several threads on the same handle: requests run one at a time.
char * systemone_decide(systemone * so, const char * request_json);

// What the model is made of, as a malloc'd JSON string: architecture, sizes, and one entry per layer with
// its kind ("shortconv", "attention", ...) and whether it belongs to the decision head. Free with
// systemone_free_string().
char * systemone_model_info(const systemone * so);

void systemone_free_string(char * s);

// The error JSON of the last systemone_decide() on this handle, "" if it succeeded. The pointer is
// valid until the next systemone_decide() on the same handle, from any thread: read it from the
// thread that got the error, before another request can run.
const char * systemone_last_error(const systemone * so);

#ifdef __cplusplus
}
#endif
