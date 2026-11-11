// The library: load a decision model, answer /v1/systemone requests without a server.
//
// The prompt of every family (what the model reads, where its answer is read from) is upstream
// llama.cpp code, unchanged: src/upstream/server-decision.cpp. This file does what the server's
// loop and its /v1/systemone handler do around it: one task per question variant, the prompt
// through the model in one batch, the scores read back as send_decision() reads them, then
// format_answer(). Parity with llama-server is the test (tests/parity.py).

#include "systemone.h"

#include "server-decision.h"
#include "llama-ext.h"
#include "llama-model.h"   // the output norm and matrix, the layer kinds: for the layer readout and model info
#include "head-lens.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

const char * decision_type_name(common_decision_type type) {
    switch (type) {
        case COMMON_DECISION_TYPE_NONE:            return "none";
        case COMMON_DECISION_TYPE_OPENJEV:         return "openjev";
        case COMMON_DECISION_TYPE_GRANITE_SWITCH:  return "granite_switch";
        case COMMON_DECISION_TYPE_LEV:             return "lev";
        case COMMON_DECISION_TYPE_KEV:             return "kev";
        case COMMON_DECISION_TYPE_NIMBLE:          return "nimble";
        case COMMON_DECISION_TYPE_LAYA:            return "laya";
        case COMMON_DECISION_TYPE_CLEF:            return "clef";
        case COMMON_DECISION_TYPE_PPLX_DECIDER:    return "pplx_decider";
        case COMMON_DECISION_TYPE_VLLM_DECISION2:  return "vllm_decision2";
        case COMMON_DECISION_TYPE_VLLM_D3:         return "vllm_d3";
        case COMMON_DECISION_TYPE_LFM2_D1:         return "lfm2_d1";
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI:    return "lfm2_d1_omni";
        case COMMON_DECISION_TYPE_H2O_LIGHTNING:   return "h2o_lightning";
        case COMMON_DECISION_TYPE_MINICPM_OCR_JEV: return "minicpm_ocr_jev";
        case COMMON_DECISION_TYPE_LFM3_D1:         return "lfm3_d1";
        case COMMON_DECISION_TYPE_LFM3_D1_OMNI:    return "lfm3_d1_omni";
        default:                                   return "unknown";
    }
}

void quiet_log(ggml_log_level, const char *, void *) {}

// On a phone the cores differ (big.LITTLE). A forward pass read in pieces lets the worker threads sleep between
// pieces, and Android wakes them on the small cores: three times slower. Pin the pool to the fastest cores,
// by cpuinfo_max_freq; on a machine where every core is the same this does nothing.
void pin_to_fast_cores(ggml_threadpool_params & tpp, int n_threads) {
    std::vector<std::pair<long, int>> freq; // (max freq, cpu)
    for (int cpu = 0; cpu < GGML_MAX_N_THREADS; cpu++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
        FILE * f = fopen(path, "r");
        if (!f) {
            break;
        }
        long v = 0;
        if (fscanf(f, "%ld", &v) == 1) {
            freq.emplace_back(v, cpu);
        }
        fclose(f);
    }
    if ((int) freq.size() < n_threads + 1) {
        return;
    }
    const long lo = std::min_element(freq.begin(), freq.end())->first;
    const long hi = std::max_element(freq.begin(), freq.end())->first;
    if (hi <= lo * 11 / 10) {
        return; // homogeneous: let the scheduler be
    }
    std::stable_sort(freq.begin(), freq.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
    for (int i = 0; i < GGML_MAX_N_THREADS; i++) {
        tpp.cpumask[i] = false;
    }
    for (int i = 0; i < n_threads && i < (int) freq.size(); i++) {
        tpp.cpumask[freq[i].second] = true;
    }
    tpp.strict_cpu     = true;
}

// one row of a weight tensor as floats, whatever its type and wherever it lives
void tensor_row_to_float(const ggml_tensor * t, int64_t row, std::vector<float> & out) {
    const int64_t n = t->ne[0];
    out.resize(n);
    const size_t row_bytes = ggml_row_size(t->type, n);
    std::vector<uint8_t> buf(row_bytes);
    ggml_backend_tensor_get(t, buf.data(), (size_t) row * t->nb[1], row_bytes);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), buf.data(), n * sizeof(float));
        return;
    }
    const auto * traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) {
        throw std::runtime_error(std::string("cannot read a ") + ggml_type_name(t->type) + " tensor");
    }
    traits->to_float(buf.data(), out.data(), n);
}

std::string error_json(const std::string & message, const char * type) {
    json err = json::object();
    err["message"] = message;
    err["type"]    = type;
    json out = json::object();
    out["error"] = err;
    return out.dump();
}

} // namespace

struct systemone {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    mtmd_context  * mctx  = nullptr; // the projector of a multimodal model, null = text only
    double media_encode_ms = 0;      // the vision encoder, last request
    // the embeddings of the last pictures by content hash: several questions about one picture encode it once
    std::unordered_map<std::string, std::vector<float>> media_cache;
    server_decision_context decision;
    std::string model_name;
    std::string type_name;
    std::string last_error;
    uint32_t n_ctx = 0;
    std::mutex mutex; // one request at a time: the context is shared

    // Layer readout (a logit lens): the residual stream at the last prompt position after each layer,
    // read through the model's own output norm and output matrix, restricted to the label tokens.
    bool lens_active = false;
    std::vector<std::vector<float>> lens_resid;                     // [n_layer][n_pos * n_embd], empty if the layer was not seen
    std::vector<int32_t> lens_pos;                                  // the prompt positions read (the last one, or the option markers)
    int32_t lens_last_base = 0;                                     // layers that only hold the output rows: row = pos - base
    int32_t lens_n_tokens = 0;                                      // rows of a full layer; fewer rows = the output rows only
    std::vector<float> lens_norm_w;                                 // the output norm weight
    const ggml_tensor * lens_out = nullptr;                         // the output matrix (or the tied embedding)
    std::unique_ptr<lfm2_head_lens> head_lens;                      // LFM2 d1: the decision head, re-run on every encoder block
    bool head_lens_tried = false;
    int n_threads = 4;
    ggml_threadpool * threadpool = nullptr;                         // one pool for llama.cpp and the readouts: no thread spawn per graph piece
    std::chrono::steady_clock::time_point lens_t0;                  // when the forward pass started
    std::vector<double> lens_ms;                                    // [n_layer] when each layer's output was ready, ms after t0
    double lens_process_ms = 0, lens_readout_ms = 0;                // the forward pass, and the lens arithmetic after it
    // per token, for prompts up to LENS_MAX_TOKENS: the whole residual stream after each layer, the input embeddings,
    // and the attention of the last position (only without flash attention, where the softmax exists as a tensor)
    static constexpr int32_t LENS_MAX_TOKENS = 512;
    bool lens_tokens = false;
    bool lens_bert = false;                                         // ModernBERT: no l_out, the block outputs are derived
    std::vector<float> lens_kqv;                                    // ModernBERT: kqv_out of the layer being computed
    std::vector<float> dbg_head_in;                                 // debug: the head's own input, derived from its first block
    std::vector<std::vector<float>> lens_full;                      // [n_layer][n_tokens * n_embd]
    std::vector<bool> lens_full_normed;                             // [n_layer] true if the row holds the output norm's result
    std::vector<float> lens_inp;                                    // [n_tokens * n_embd]
    std::unordered_map<int, std::vector<std::vector<float>>> lens_attn; // [il] -> [head][n_kv], the last query row of every head

    // ggml asks before each node whether its data is needed, then calls again once it is computed
    static bool cb_eval(ggml_tensor * t, bool ask, void * user_data) {
        auto * so = static_cast<systemone *>(user_data);
        if (!so->lens_active) {
            return !ask;
        }
        const char * name = ggml_get_name(t);
        if (so->lens_bert) {
            return so->cb_eval_bert(t, ask, name);
        }
        const bool is_l_out = strncmp(name, "l_out-", 6) == 0;
        const bool is_inp   = so->lens_tokens && strcmp(name, "inp_embd") == 0;
        const bool is_attn  = so->lens_tokens && strncmp(name, "kq_soft_max-", 12) == 0;
        if (!is_l_out && !is_inp && !is_attn) {
            return !ask;
        }
        if (ask) {
            return true;
        }
        if (t->type != GGML_TYPE_F32) {
            return true;
        }
        if (is_inp) {
            so->lens_inp.resize(t->ne[0] * t->ne[1]);
            ggml_backend_tensor_get(t, so->lens_inp.data(), 0, so->lens_inp.size() * sizeof(float));
            return true;
        }
        if (is_attn) {
            // [n_kv, n_q, n_head]: the last query row of every head, averaged
            const int il = atoi(name + 12);
            const int64_t n_kv = t->ne[0], n_q = t->ne[1], n_head = t->ne[2];
            if (n_q < 1 || n_head < 1) {
                return true;
            }
            std::vector<std::vector<float>> heads(n_head, std::vector<float>(n_kv));
            for (int64_t h = 0; h < n_head; h++) {
                ggml_backend_tensor_get(t, heads[h].data(), (size_t) (n_q - 1) * t->nb[1] + (size_t) h * t->nb[2], n_kv * sizeof(float));
            }
            so->lens_attn[il] = std::move(heads);
            return true;
        }
        const int il = atoi(name + 6);
        if (il < 0 || il >= (int) so->lens_resid.size() || t->ne[1] < 1) {
            return true;
        }
        // the last layer, and in a decision head every layer, keeps only the output positions
        const bool reduced = t->ne[1] < so->lens_n_tokens;
        const int64_t n_embd = t->ne[0];
        auto & rows = so->lens_resid[il];
        rows.assign(so->lens_pos.size() * n_embd, 0.0f);
        bool ok = true;
        for (size_t k = 0; k < so->lens_pos.size(); k++) {
            const int64_t row = reduced ? so->lens_pos[k] - so->lens_last_base : so->lens_pos[k];
            if (row < 0 || row >= t->ne[1]) {
                ok = false;
                break;
            }
            ggml_backend_tensor_get(t, rows.data() + k * n_embd, (size_t) row * t->nb[1], n_embd * sizeof(float));
        }
        if (!ok) {
            rows.clear();
        }
        so->lens_ms[il] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - so->lens_t0).count();
        if (so->lens_tokens && !reduced && il < (int) so->lens_full.size()) {
            auto & full = so->lens_full[il];
            full.resize(t->ne[0] * t->ne[1]);
            ggml_backend_tensor_get(t, full.data(), 0, full.size() * sizeof(float));
        }
        return true;
    }

    // ModernBERT names no block output. The residual after block il is ffn_inp(il+1) - kqv_out(il+1); the last
    // encoder block's is only seen after the output norm, as final_norm_out, which the head takes as it is.
    bool cb_eval_bert(ggml_tensor * t, bool ask, const char * name) {
        const bool is_kqv  = strncmp(name, "kqv_out-", 8) == 0;
        const bool is_ffn  = strncmp(name, "ffn_inp-", 8) == 0;
        const bool is_fin  = strcmp(name, "final_norm_out") == 0;
        const bool is_inp  = strcmp(name, "inp_norm-0") == 0;
        const bool is_attn = strncmp(name, "kq_soft_max-", 12) == 0;
        if (!is_kqv && !is_ffn && !is_fin && !is_inp && !is_attn) {
            return !ask;
        }
        if (ask) {
            return true;
        }
        if (t->type != GGML_TYPE_F32) {
            return true;
        }
        const int n_enc = (int) lens_full.size() - (int) model->hparams.n_layer_decision; // the head's blocks derive nothing
        const size_t n = (size_t) t->ne[0] * t->ne[1];
        if (is_inp) {
            lens_inp.resize(n);
            ggml_backend_tensor_get(t, lens_inp.data(), 0, n * sizeof(float));
            return true;
        }
        if (is_attn) {
            const int il = atoi(name + 12);
            const int64_t n_kv = t->ne[0], n_q = t->ne[1], n_head = t->ne[2];
            if (n_q < 1 || n_head < 1) return true;
            std::vector<std::vector<float>> heads(n_head, std::vector<float>(n_kv));
            for (int64_t h = 0; h < n_head; h++) {
                ggml_backend_tensor_get(t, heads[h].data(), (size_t) (n_q - 1) * t->nb[1] + (size_t) h * t->nb[2], n_kv * sizeof(float));
            }
            lens_attn[il] = std::move(heads);
            return true;
        }
        if (is_fin) {
            if (n_enc > 0 && (int64_t) lens_n_tokens == t->ne[1]) {
                auto & out = lens_full[n_enc - 1];
                out.resize(n);
                ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
                lens_full_normed[n_enc - 1] = true;
                lens_ms[n_enc - 1] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lens_t0).count();
            }
            return true;
        }
        const int il = atoi(name + 8);
        if (il == n_enc && (int64_t) lens_n_tokens == t->ne[1] && getenv("SYSTEMONE_DUMP") != nullptr) {
            // debug: the first head block's input = final_norm_out + type_row(0); derive it once
            if (is_kqv) {
                lens_kqv.resize(n);
                ggml_backend_tensor_get(t, lens_kqv.data(), 0, n * sizeof(float));
            } else if (dbg_head_in.empty() && lens_kqv.size() == n) {
                dbg_head_in.resize(n);
                ggml_backend_tensor_get(t, dbg_head_in.data(), 0, n * sizeof(float));
                for (size_t i = 0; i < n; i++) dbg_head_in[i] -= lens_kqv[i];
            }
            return true;
        }
        if (il < 1 || il >= n_enc || (int64_t) lens_n_tokens != t->ne[1]) {
            return true; // layer 0 has no predecessor to derive; reduced layers are the head's
        }
        if (is_kqv) {
            lens_kqv.resize(n);
            ggml_backend_tensor_get(t, lens_kqv.data(), 0, n * sizeof(float));
            return true;
        }
        if (lens_kqv.size() != n) {
            return true;
        }
        auto & out = lens_full[il - 1];
        out.resize(n);
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
        for (size_t i = 0; i < n; i++) {
            out[i] -= lens_kqv[i];
        }
        lens_ms[il - 1] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lens_t0).count();
        return true;
    }

    // the output norm and matrix, once; throws if this architecture's head is not RMS norm + matrix
    void lens_init() {
        if (lens_out) {
            return;
        }
        if (!model->output_norm || model->output_norm_b || model->output_s) {
            throw std::runtime_error("layer readout: this model's output head is not a norm and a matrix");
        }
        tensor_row_to_float(model->output_norm, 0, lens_norm_w);
        lens_out = model->output ? model->output : model->tok_embd; // may be null for a model read from embeddings
    }

    // the logits of many normalised residuals (one per layer) through the output matrix in one ggml mul_mat, which
    // works whether or not llama.cpp repacked the matrix for its fast kernels (reading its rows would not).
    // Returns [n_rows][n_vocab].
    std::vector<std::vector<float>> lens_logits(const std::vector<std::vector<float>> & hs) {
        if (!lens_out) {
            throw std::runtime_error("layer readout: the model has no output matrix");
        }
        const int64_t n_vocab = lens_out->ne[1];
        const int64_t n_embd  = lens_out->ne[0];
        const int64_t n_rows  = (int64_t) hs.size();
        ggml_init_params ip = { (size_t) n_rows * (n_embd + n_vocab) * sizeof(float) + 4 * 1024 * 1024, nullptr, false };
        ggml_context * ctx_l = ggml_init(ip);
        if (!ctx_l) {
            throw std::runtime_error("layer readout: out of memory");
        }
        ggml_tensor * x = ggml_new_tensor_2d(ctx_l, GGML_TYPE_F32, n_embd, n_rows);
        for (int64_t r = 0; r < n_rows; r++) {
            if ((int64_t) hs[r].size() != n_embd) {
                ggml_free(ctx_l);
                throw std::runtime_error("layer readout: residual width does not match the output matrix");
            }
            memcpy((float *) x->data + r * n_embd, hs[r].data(), n_embd * sizeof(float));
        }
        ggml_tensor * y = ggml_mul_mat(ctx_l, const_cast<ggml_tensor *>(lens_out), x); // [n_vocab, n_rows]
        ggml_cgraph * gf = ggml_new_graph(ctx_l);
        ggml_build_forward_expand(gf, y);
        // the work buffer outside the context: its size depends on the threads and the kernels
        ggml_cplan plan = ggml_graph_plan(gf, n_threads, threadpool);
        std::vector<uint8_t> work(plan.work_size + 64);
        plan.work_data = work.data();
        const ggml_status st = ggml_graph_compute(gf, &plan);
        if (st != GGML_STATUS_SUCCESS) {
            ggml_free(ctx_l);
            throw std::runtime_error("layer readout: compute failed");
        }
        std::vector<std::vector<float>> logits(n_rows);
        for (int64_t r = 0; r < n_rows; r++) {
            const float * row = (const float *) ((const char *) y->data + r * y->nb[1]);
            logits[r].assign(row, row + n_vocab);
        }
        ggml_free(ctx_l);
        return logits;
    }

    // the output norm of the model applied to one residual row: what the head reads
    std::vector<float> lens_norm(const float * resid, size_t n) const {
        if (n != lens_norm_w.size()) {
            throw std::runtime_error("layer readout: residual width does not match the output norm");
        }
        double ss = 0.0;
        for (size_t i = 0; i < n; i++) {
            ss += (double) resid[i] * resid[i];
        }
        const float scale = 1.0f / sqrtf((float) (ss / n) + model->hparams.f_norm_rms_eps);
        std::vector<float> h(n);
        for (size_t i = 0; i < n; i++) {
            h[i] = resid[i] * scale * lens_norm_w[i];
        }
        return h;
    }

    // the output-embedding projection some models apply after the norm (qwen35 / kev: cls_out, [n_embd -> n_embd_out]),
    // on many rows at once, in place through ggml_mul_mat. Without such a projection the rows pass through.
    std::vector<std::vector<float>> lens_project(const std::vector<std::vector<float>> & hs) {
        const ggml_tensor * w = model->cls_out;
        if (!w) {
            return hs;
        }
        const int64_t n_in = w->ne[0], n_out = w->ne[1], n_rows = (int64_t) hs.size();
        ggml_init_params ip = { (size_t) n_rows * (n_in + n_out) * sizeof(float) + (size_t) n_out * sizeof(float) + 4 * 1024 * 1024, nullptr, false };
        ggml_context * ctx_p = ggml_init(ip);
        if (!ctx_p) {
            throw std::runtime_error("layer readout: out of memory");
        }
        ggml_tensor * x = ggml_new_tensor_2d(ctx_p, GGML_TYPE_F32, n_in, n_rows);
        for (int64_t r = 0; r < n_rows; r++) {
            if ((int64_t) hs[r].size() != n_in) {
                ggml_free(ctx_p);
                throw std::runtime_error("layer readout: residual width does not match the projection");
            }
            memcpy((float *) x->data + r * n_in, hs[r].data(), n_in * sizeof(float));
        }
        ggml_tensor * y = ggml_mul_mat(ctx_p, const_cast<ggml_tensor *>(w), x);
        if (model->cls_out_b) {
            std::vector<float> b;
            tensor_row_to_float(model->cls_out_b, 0, b);
            ggml_tensor * bt = ggml_new_tensor_1d(ctx_p, GGML_TYPE_F32, n_out);
            memcpy(bt->data, b.data(), n_out * sizeof(float));
            y = ggml_add(ctx_p, y, bt);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx_p);
        ggml_build_forward_expand(gf, y);
        ggml_cplan plan = ggml_graph_plan(gf, n_threads, threadpool);
        std::vector<uint8_t> work(plan.work_size + 64);
        plan.work_data = work.data();
        if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
            ggml_free(ctx_p);
            throw std::runtime_error("layer readout: compute failed");
        }
        std::vector<std::vector<float>> out(n_rows);
        for (int64_t r = 0; r < n_rows; r++) {
            const float * row = (const float *) ((const char *) y->data + r * y->nb[1]);
            out[r].assign(row, row + n_out);
        }
        ggml_free(ctx_p);
        return out;
    }

    // the label logits every layer's residual would give through the output head: the norm, then the matrix,
    // all layers in one matmul. Empty residuals give empty rows.
    std::vector<std::vector<float>> lens_scores_all(const std::vector<std::vector<float>> & resids, const std::vector<llama_token> & labels) {
        std::vector<std::vector<float>> hs;
        std::vector<size_t> which;
        for (size_t il = 0; il < resids.size(); il++) {
            if (!resids[il].empty()) {
                hs.push_back(lens_norm(resids[il].data(), resids[il].size()));
                which.push_back(il);
            }
        }
        std::vector<std::vector<float>> out(resids.size());
        if (hs.empty()) {
            return out;
        }
        const auto logits = lens_logits(hs);
        for (size_t k = 0; k < which.size(); k++) {
            std::vector<float> scores;
            for (const llama_token label : labels) {
                if (label < 0 || (size_t) label >= logits[k].size()) {
                    throw std::runtime_error("label token out of the vocabulary");
                }
                scores.push_back(logits[k][label]);
            }
            out[which[k]] = std::move(scores);
        }
        return out;
    }

    // what the per-token capture of one task turns into
    struct lens_tokens_result {
        std::vector<std::string> pieces;                       // the prompt as the model reads it
        std::vector<std::vector<float>> change;                // [layer][token] 0..1, how much the layer moved the token
        std::unordered_map<int, std::vector<std::vector<float>>> attention; // [layer][head][token], where the last position looked
    };

    // ||after - before|| per token for each layer, scaled to the layer's largest move
    lens_tokens_result lens_token_readout(const llama_tokens & tokens) {
        lens_tokens_result out;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        for (const llama_token t : tokens) {
            out.pieces.push_back(t == LLAMA_TOKEN_NULL ? std::string("\xE2\x96\xA7") : common_token_to_piece(vocab, t, true)); // a media position: one square
        }
        const int64_t n_tokens = (int64_t) tokens.size();
        const std::vector<float> * prev = lens_inp.empty() ? nullptr : &lens_inp;
        for (const auto & full : lens_full) {
            std::vector<float> change(n_tokens, 0.0f);
            if (!full.empty() && prev && prev->size() == full.size() && full.size() % n_tokens == 0) {
                const int64_t n_embd = (int64_t) full.size() / n_tokens;
                float max_v = 0.0f;
                for (int64_t t = 0; t < n_tokens; t++) {
                    double ss = 0.0;
                    for (int64_t i = 0; i < n_embd; i++) {
                        const double d = (double) full[t * n_embd + i] - (double) (*prev)[t * n_embd + i];
                        ss += d * d;
                    }
                    change[t] = (float) sqrt(ss);
                    max_v = std::max(max_v, change[t]);
                }
                if (max_v > 0) {
                    for (auto & v : change) v /= max_v;
                }
            }
            out.change.push_back(std::move(change));
            if (!full.empty()) prev = &full;
        }
        for (auto & [il, heads] : lens_attn) {
            if (!heads.empty() && (int64_t) heads[0].size() == n_tokens) {
                out.attention[il] = heads;
            }
        }
        return out;
    }

    static std::vector<float> pool_groups(std::vector<float> scores, const std::vector<int32_t> & groups) {
        if (groups.empty()) {
            return scores;
        }
        std::vector<float> pooled;
        size_t i = 0;
        for (const int32_t n : groups) {
            if (n <= 0 || i + n > scores.size()) {
                throw std::runtime_error("label groups do not match the labels");
            }
            pooled.push_back(*std::max_element(scores.begin() + i, scores.begin() + i + n));
            i += n;
        }
        return pooled;
    }

    // what the model is made of, for an architecture view
    json model_info() const {
        const auto & hp = model->hparams;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        char desc[256];
        llama_model_desc(model, desc, sizeof(desc));
        json info = json::object();
        info["model"]            = model_name;
        info["name"]             = model->name;
        info["architecture"]     = model->arch_name();
        info["description"]      = desc;
        info["decision_type"]    = type_name;
        info["n_params"]         = (int64_t) llama_model_n_params(model);
        info["size_bytes"]       = (int64_t) llama_model_size(model);
        info["n_layer"]          = (int32_t) hp.n_layer();
        info["n_layer_decision"] = (int32_t) hp.n_layer_decision;
        info["n_embd"]           = (int32_t) hp.n_embd;
        info["n_head"]           = llama_model_n_head(model);
        info["n_head_kv"]        = llama_model_n_head_kv(model);
        info["n_vocab"]          = llama_vocab_n_tokens(vocab);
        info["n_ctx_train"]      = llama_model_n_ctx_train(model);
        info["n_ctx"]            = (int32_t) n_ctx;
        const bool lfm = model->arch == LLM_ARCH_LFM2 || model->arch == LLM_ARCH_LFM3;
        // bytes of each block's weights in memory, and of everything outside the blocks (embeddings, output)
        std::vector<int64_t> block_bytes(hp.n_layer(), 0);
        int64_t other_bytes = 0;
        for (const auto & [name, t] : model->tensors_by_name) {
            if (!t) continue;
            const int64_t bytes = (int64_t) ggml_nbytes(t);
            if (name.rfind("blk.", 0) == 0) {
                const uint32_t il = (uint32_t) atoi(name.c_str() + 4);
                if (il < hp.n_layer()) { block_bytes[il] += bytes; continue; }
            }
            other_bytes += bytes;
        }
        info["bytes_outside_blocks"] = other_bytes;
        json layers = json::array();
        for (uint32_t il = 0; il < hp.n_layer(); il++) {
            json l = json::object();
            l["i"]     = (int32_t) il;
            l["bytes"] = block_bytes[il];
            if (hp.is_recr(il)) {
                l["kind"] = lfm ? "shortconv" : "recurrent";
            } else {
                l["kind"] = hp.is_swa(il) ? "attention_swa" : "attention";
            }
            l["decision_head"] = hp.n_layer_decision > 0 && il >= hp.n_layer() - hp.n_layer_decision;
            layers.push_back(l);
        }
        info["layers"] = layers;
        return info;
    }

    ~systemone() {
        if (mctx)  mtmd_free(mctx);
        if (ctx)   llama_free(ctx);
        if (model) llama_model_free(model);
        if (threadpool) ggml_threadpool_free(threadpool);
    }

    // The scores of one task: the prompt through the model in one batch, read as the server reads it.
    // layers, if given, receives the label scores of every layer (empty rows for layers without a readout)
    std::vector<float> run(const server_task & task, std::vector<std::vector<float>> * layers = nullptr, std::vector<double> * layer_ms = nullptr,
                           lens_tokens_result * per_token = nullptr) {
        const llama_tokens & tokens = task.tokens.tokens;
        const int32_t n_tokens = (int32_t) tokens.size();
        if (n_tokens == 0) {
            throw std::runtime_error("empty prompt");
        }
        if ((uint32_t) n_tokens > n_ctx) {
            throw std::runtime_error(string_format(
                "request (%d tokens) exceeds the context size (%u tokens), load with a larger n_ctx", n_tokens, n_ctx));
        }

        const auto & d = task.decision;
        const bool wants_embd = d.pooled || !d.markers.empty() || d.pointer >= 0 || d.n_scores > 0 || !d.order.empty();
        const int32_t pos_first = d.pos_first();

        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_clear(mem, true);
        }
        llama_set_embeddings(ctx, wants_embd);

        llama_batch_ext * batch = llama_batch_ext_init(ctx);
        std::unique_ptr<llama_batch_ext, void (*)(llama_batch_ext *)> batch_guard(batch, llama_batch_ext_free);

        // Media chunks (an image) are encoded by the projector first; their embeddings then sit in the same batch
        // as the text tokens, one entry per position: what llama-server does for a model that reads its prompt in
        // one batch (add_prompt_mixed). The encoder outputs must outlive llama_process().
        std::unordered_map<int32_t, const float *> media_embd; // first token index of a chunk -> its embeddings
        media_encode_ms = 0;
        if (task.tokens.has_mtmd && !task.tokens.map_idx_to_media.empty()) {
            if (!mctx) {
                throw std::runtime_error("this model was loaded without a projector (mmproj), it cannot read images");
            }
            const auto t_enc = std::chrono::steady_clock::now();
            const size_t n_in = (size_t) llama_model_n_embd_inp(model);
            for (const auto & [start, chunk] : task.tokens.map_idx_to_media) {
                const char * cid = mtmd_input_chunk_get_id(chunk.get());
                const std::string id = cid ? cid : "";
                const size_t n_floats = mtmd_input_chunk_get_n_tokens(chunk.get()) * n_in;
                auto hit = id.empty() ? media_cache.end() : media_cache.find(id);
                if (hit == media_cache.end() || hit->second.size() != n_floats) {
                    mtmd::batch_ptr mb(mtmd_batch_init(mctx));
                    if (mtmd_batch_add_chunk(mb.get(), chunk.get()) != 0 || mtmd_batch_encode(mb.get()) != 0) {
                        throw std::runtime_error("failed to encode the image");
                    }
                    const float * embd = mtmd_batch_get_output_embd(mb.get(), chunk.get());
                    if (!embd) {
                        throw std::runtime_error("the image encoder returned nothing");
                    }
                    if (media_cache.size() >= 4) {
                        media_cache.clear();
                    }
                    hit = media_cache.insert_or_assign(id.empty() ? std::to_string(start) : id, std::vector<float>(embd, embd + n_floats)).first;
                }
                media_embd[(int32_t) start] = hit->second.data();
            }
            media_encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc).count();
        }
        const size_t n_embd_inp = (size_t) llama_model_n_embd_inp(model);
        const bool   is_mrope   = mctx && mtmd_decode_use_mrope(mctx);

        // row in the batch of each prompt position that has an output
        std::vector<int32_t> rows(n_tokens, -1);
        auto set_outputs = [&](int32_t i, int32_t idx) {
            rows[i] = idx;
            const bool last = i == n_tokens - 1;
            if (!d.labels.empty() && last) {
                llama_batch_ext_set_output_logits(batch, idx, true);
            }
            if (wants_embd && (d.pooled || (pos_first >= 0 && i >= pos_first) || (d.n_scores > 0 && i < d.n_scores))) {
                llama_batch_ext_set_output_embd(batch, idx, true);
            }
            if (!d.order.empty()) {
                llama_batch_ext_set_decision_order(batch, idx, (llama_decision_order) d.order[i]);
            }
        };
        llama_pos pos_next = 0;
        for (int32_t i = 0; i < n_tokens; i++) {
            if (tokens[i] != LLAMA_TOKEN_NULL) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
                if (idx < 0) {
                    throw std::runtime_error(string_format("the prompt (%d tokens) does not fit in one batch", n_tokens));
                }
                const llama_pos pos = pos_next++; // text tokens: one position each, nothing is set by default
                llama_batch_ext_set_pos(batch, idx, &pos);
                set_outputs(i, idx);
                continue;
            }
            // a media chunk: n entries, one embedding row each
            const auto it = media_embd.find(i);
            if (it == media_embd.end()) {
                throw std::runtime_error("unexpected layout of the media in the prompt");
            }
            const mtmd_input_chunk * chunk = task.tokens.find_chunk(i).get();
            const auto * image   = mtmd_input_chunk_get_tokens_image(chunk);
            const size_t n_chunk = mtmd_input_chunk_get_n_tokens(chunk);
            const llama_pos pos_0 = pos_next;
            for (size_t k = 0; k < n_chunk; k++) {
                const int32_t idx = llama_batch_ext_add_embd(batch, 0, { it->second + k * n_embd_inp, 1, n_embd_inp });
                if (idx < 0) {
                    throw std::runtime_error(string_format("the prompt (%d positions, an image among them) does not fit in one batch", n_tokens));
                }
                const llama_pos p = pos_0 + (llama_pos) k;
                llama_pos pos[4] = { p, p, p, p };
                if (is_mrope && image) {
                    const mtmd_decoder_pos rel = mtmd_image_tokens_get_decoder_pos(image, pos_0, k);
                    pos[0] = rel.t; pos[1] = rel.y; pos[2] = rel.x; pos[3] = rel.z;
                }
                llama_batch_ext_set_pos(batch, idx, pos);
                set_outputs(i + (int32_t) k, idx);
            }
            pos_next = pos_0 + (llama_pos) n_chunk;
            i += (int32_t) n_chunk - 1;
        }

        // the layer readout: label logits at the last position, or one embedding column at each option's marker
        const bool lens_labels  = !d.labels.empty() && !d.pooled;
        // a model with a trained decision head (LFM2 d1-omni) answers through that head, not through the output norm:
        // the readout re-runs the head on the residual after every encoder block (head-lens.cpp)
        const bool marker_readout = d.labels.empty() && !d.markers.empty() && d.pointer < 0 && !d.pooled && d.n_scores == 0;
        const bool lens_markers = marker_readout && model->hparams.n_layer_decision == 0;
        // pointer readout (kev): the embedding at the question's position dotted with each option's, two halves
        const bool lens_pointer = d.labels.empty() && !d.markers.empty() && d.pointer >= 0 && !d.pooled && d.n_scores == 0 &&
                                  model->hparams.n_layer_decision == 0 &&
                                  (model->cls_out ? model->cls_out->ne[1] == llama_model_n_embd_out(model)
                                                  : llama_model_n_embd_out(model) == llama_model_n_embd(model));
        bool lens_head = false;
        const bool bert = model->arch == LLM_ARCH_MODERN_BERT;
        if (layers != nullptr && marker_readout && model->hparams.n_layer_decision > 0 && n_tokens <= LENS_MAX_TOKENS) {
            if (!head_lens_tried) {
                head_lens_tried = true;
                auto h = std::make_unique<lfm2_head_lens>();
                if (h->init(model)) {
                    head_lens = std::move(h);
                }
            }
            lens_head = head_lens != nullptr;
        }
        const bool lens = layers != nullptr && (lens_labels || lens_markers || lens_head || lens_pointer);
        if (lens) {
            lens_init();
            lens_pos = lens_labels ? std::vector<int32_t>{n_tokens - 1} : d.markers;
            if (lens_pointer) {
                lens_pos.push_back(d.pointer); // the question's position last
            }
            lens_n_tokens = n_tokens;
            lens_last_base = n_tokens - 1;
            for (int32_t i = 0; i < n_tokens; i++) { // the first position with an output: row 0 of the last layer
                if ((!d.labels.empty() && i == n_tokens - 1) ||
                    (wants_embd && (d.pooled || (pos_first >= 0 && i >= pos_first) || (d.n_scores > 0 && i < d.n_scores)))) {
                    lens_last_base = i;
                    break;
                }
            }
            lens_resid.assign(llama_model_n_layer(model), {});
            lens_ms.assign(llama_model_n_layer(model), -1.0);
            // per-token capture only within a budget: all blocks times all tokens times the width, in floats
            const size_t budget_floats = 64u * 1024u * 1024u / sizeof(float);
            const size_t need_floats   = (size_t) llama_model_n_layer(model) * n_tokens * llama_model_n_embd(model);
            lens_tokens = (per_token != nullptr || lens_head) && n_tokens <= LENS_MAX_TOKENS && need_floats <= budget_floats;
            lens_bert = bert && lens_tokens;
            lens_full.assign(lens_tokens ? llama_model_n_layer(model) : 0, {});
            lens_full_normed.assign(lens_full.size(), false);
            lens_kqv.clear();
            lens_inp.clear();
            lens_attn.clear();
            lens_t0 = std::chrono::steady_clock::now();
            lens_active = true;
        }
        const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        lens_active = false;
        lens_bert = false;
        lens_kqv.clear();
        if (lens) {
            lens_process_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lens_t0).count();
            if (layer_ms) {
                *layer_ms = lens_ms;
            }
        }
        if (rc != 0) {
            throw std::runtime_error(string_format("llama_process failed (%d)", rc));
        }
        if (lens) {
            const auto t_readout = std::chrono::steady_clock::now();
            layers->clear();
            const int n_layer_enc = (int) llama_model_n_layer(model) - (int) model->hparams.n_layer_decision;
            const auto label_scores = lens_labels ? lens_scores_all(lens_resid, d.labels) : std::vector<std::vector<float>>();
            // pointer: every layer's rows (markers, then the pointer) normalised and projected in one go, then
            // as send_decision() reads it: q = the first half at the pointer, k = the second half at each marker
            std::vector<std::vector<float>> pointer_scores(lens_resid.size());
            if (lens_pointer) {
                const size_t n_pos = lens_pos.size();
                std::vector<std::vector<float>> hs;
                std::vector<size_t> which;
                for (size_t il = 0; il < lens_resid.size(); il++) {
                    const auto & r = lens_resid[il];
                    if (r.empty()) continue;
                    const size_t n_embd = r.size() / n_pos;
                    for (size_t k = 0; k < n_pos; k++) {
                        hs.push_back(lens_norm(r.data() + k * n_embd, n_embd));
                    }
                    which.push_back(il);
                }
                if (!hs.empty()) {
                    const auto es = lens_project(hs);
                    const size_t n_out = es[0].size(), n_ptr = n_out / 2;
                    for (size_t j = 0; j < which.size(); j++) {
                        const auto & eq = es[j * n_pos + n_pos - 1];
                        std::vector<float> scores;
                        for (size_t k = 0; k + 1 < n_pos; k++) {
                            const auto & ek = es[j * n_pos + k];
                            double dot = 0.0;
                            for (size_t i = 0; i < n_ptr; i++) {
                                dot += (double) eq[i] * ek[n_ptr + i];
                            }
                            scores.push_back((float) (dot / sqrt((double) n_ptr)));
                        }
                        pointer_scores[which[j]] = std::move(scores);
                    }
                }
            }
            for (size_t il = 0; il < lens_resid.size(); il++) {
                const auto & resid = lens_resid[il];
                if (lens_head) {
                    // the head on this encoder block's full residual stream; the head's own blocks have no readout yet
                    const auto & full = il < lens_full.size() ? lens_full[il] : std::vector<float>();
                    if ((int) il < n_layer_enc && !full.empty() && full.size() == (size_t) n_tokens * llama_model_n_embd(model)) {
                        layers->push_back(head_lens->run(full.data(), n_tokens, d.markers, d.column, n_threads, lens_full_normed[il], threadpool));
                    } else {
                        layers->push_back({});
                    }
                } else if (resid.empty()) {
                    layers->push_back({});
                } else if (lens_labels) {
                    layers->push_back(pool_groups(label_scores[il], d.label_groups));
                } else if (lens_pointer) {
                    layers->push_back(pointer_scores[il]);
                } else {
                    // markers: the embedding column of each option, as send_decision() reads it from the final embeddings
                    const size_t n_embd = resid.size() / lens_pos.size();
                    std::vector<float> scores;
                    for (size_t k = 0; k < lens_pos.size(); k++) {
                        const std::vector<float> h = lens_norm(resid.data() + k * n_embd, n_embd);
                        if (d.column < 0 || (size_t) d.column >= h.size()) {
                            throw std::runtime_error("embedding column out of range");
                        }
                        scores.push_back(h[d.column]);
                    }
                    layers->push_back(std::move(scores));
                }
            }
            lens_resid.clear();
            if (lens_tokens && per_token) {
                *per_token = lens_token_readout(tokens);
            }
            if (getenv("SYSTEMONE_DUMP") != nullptr && !dbg_head_in.empty() && lens_head) {
                const int n_enc_dbg = (int) llama_model_n_layer(model) - (int) model->hparams.n_layer_decision;
                const auto & fin = lens_full[n_enc_dbg - 1];
                const auto tr = head_lens->type_row(0);
                const size_t ne = tr.size();
                double max_d = 0, max_v = 0;
                for (size_t i = 0; i < fin.size() && i < dbg_head_in.size(); i++) {
                    const double expect = dbg_head_in[i] - tr[i % ne];
                    max_d = std::max(max_d, fabs(expect - fin[i]));
                    max_v = std::max(max_v, fabs(expect));
                }
                fprintf(stderr, "--- head input check: captured final_norm_out vs derived, max diff %.5f (max value %.3f), n %zu vs %zu\n", max_d, max_v, fin.size(), dbg_head_in.size());
                dbg_head_in.clear();
            }
            if (getenv("SYSTEMONE_DUMP") != nullptr) {
                for (size_t il = 0; il < layers->size(); il++) {
                    if ((*layers)[il].empty()) continue;
                    fprintf(stderr, "--- lens %zu:", il);
                    for (const float v : (*layers)[il]) fprintf(stderr, " %.5f", v);
                    fprintf(stderr, "\n");
                }
            }
            lens_full.clear();
            lens_inp.clear();
            lens_attn.clear();
            lens_tokens = false;
            lens_readout_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_readout).count();
        }

        std::vector<float> scores;
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool dump = getenv("SYSTEMONE_DUMP") != nullptr; // the prompt and the raw scores, to compare with llama-server
        if (dump) {
            std::string text;
            for (const llama_token t : tokens) {
                text += t == LLAMA_TOKEN_NULL ? std::string("\xE2\x96\xA7") : common_token_to_piece(vocab, t, true);
            }
            fprintf(stderr, "--- prompt (%d tokens) ---\n%s\n--- labels %zu groups %zu markers %zu pointer %d order %zu pooled %d ---\n",
                    n_tokens, text.c_str(), d.labels.size(), d.label_groups.size(), d.markers.size(), d.pointer, d.order.size(), (int) d.pooled);
        }

        if (d.pooled) {
            const float * embd = llama_get_embeddings_seq(ctx, 0);
            if (!embd) {
                throw std::runtime_error("failed to get the class scores");
            }
            scores.assign(embd, embd + llama_model_n_embd_out(model));
            return scores;
        }

        if (!d.labels.empty()) {
            const float * logits = llama_get_logits_ith(ctx, rows[n_tokens - 1]);
            if (!logits) {
                throw std::runtime_error("failed to get logits");
            }
            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            for (const llama_token label : d.labels) {
                if (label < 0 || label >= n_vocab) {
                    throw std::runtime_error("label token out of the vocabulary");
                }
                scores.push_back(logits[label]);
            }
            if (!d.label_groups.empty()) {
                std::vector<float> pooled;
                size_t i = 0;
                for (const int32_t n : d.label_groups) {
                    if (n <= 0 || i + n > scores.size()) {
                        throw std::runtime_error("label groups do not match the labels");
                    }
                    pooled.push_back(*std::max_element(scores.begin() + i, scores.begin() + i + n));
                    i += n;
                }
                scores = std::move(pooled);
            }
            if (dump) {
                fprintf(stderr, "--- raw scores:");
                for (const float v : scores) fprintf(stderr, " %.5f", v);
                fprintf(stderr, "\n");
            }
            return scores;
        }

        auto get_embd = [&](int32_t pos) -> const float * {
            return pos >= 0 && pos < n_tokens ? llama_get_embeddings_ith(ctx, rows[pos]) : nullptr;
        };

        // joint head: the scores are the first rows
        for (int32_t i = 0; i < d.n_scores; i++) {
            const float * embd = get_embd(i);
            if (!embd) {
                throw std::runtime_error("failed to get embeddings");
            }
            scores.push_back(embd[0]);
        }

        const int32_t n_embd_out = llama_model_n_embd_out(model);
        const int32_t n_pointer  = n_embd_out / 2;
        const float * embd_q = d.pointer >= 0 ? get_embd(d.pointer) : nullptr;
        if (d.column < 0 || d.column >= n_embd_out) {
            throw std::runtime_error("embedding column out of range");
        }
        if (dump) {
            fprintf(stderr, "--- marker readout: column %d, n_embd_out %d, pointer %d\n", d.column, n_embd_out, d.pointer);
        }
        for (const int32_t marker : d.markers) {
            const float * embd = get_embd(marker);
            if (!embd || (d.pointer >= 0 && !embd_q)) {
                throw std::runtime_error("failed to get embeddings at the option markers");
            }
            if (d.pointer < 0) {
                scores.push_back(embd[d.column]);
                continue;
            }
            float dot = 0.0f;
            for (int32_t i = 0; i < n_pointer; i++) {
                dot += embd_q[i] * embd[n_pointer + i];
            }
            scores.push_back(dot / sqrtf((float) n_pointer));
        }
        if (dump) {
            fprintf(stderr, "--- raw scores:");
            for (const float v : scores) fprintf(stderr, " %.5f", v);
            fprintf(stderr, "\n");
        }
        return scores;
    }

    // One request, start to end under the mutex, last_error included: two threads deciding on the same
    // handle must not write last_error at the same time.
    std::string decide(const std::string & request) {
        std::lock_guard<std::mutex> lock(mutex);
        std::string out;
        try {
            out = answer(request);
        } catch (...) {
            out = error_json("internal error", "server_error");
        }
        if (out.rfind("{\"error\"", 0) == 0) {
            last_error = out;
        } else {
            last_error.clear();
        }
        return out;
    }

private:
    // the answer of each layer in the shape of the final answer: a row of probabilities per option, or the
    // noul probability; null for a layer without a readout
    json layers_answer(const server_decision_question & question, size_t n_variants, size_t first_task,
                       const std::vector<std::vector<std::vector<float>>> & layer_results,
                       const std::vector<std::vector<double>> & layer_ms_results,
                       const std::vector<lens_tokens_result> & token_results, const std::string & error, bool has_media = false) const {
        json out = json::object();
        if (!error.empty()) {
            out["error"] = error;
            return out;
        }
        json options = json::array();
        for (const auto & option : question.options) {
            options.push_back(option.key);
        }
        out["options"] = options;
        const size_t n_layer = layer_results[first_task].size();
        json rows = json::array();
        for (size_t il = 0; il < n_layer; il++) {
            std::vector<std::vector<float>> scores;
            bool complete = true;
            for (size_t v = 0; v < n_variants; v++) {
                const auto & per_layer = layer_results[first_task + v];
                if (il >= per_layer.size() || per_layer[il].empty()) {
                    complete = false;
                    break;
                }
                scores.push_back(per_layer[il]);
            }
            if (!complete) {
                rows.push_back(nullptr);
                continue;
            }
            const json a = decision.format_answer(question, scores, has_media);
            if (a.contains("noul")) {
                rows.push_back(a.at("noul").get<double>());
            } else if (a.contains("probabilities")) {
                json row = json::array();
                for (const auto & option : question.options) {
                    row.push_back(a.at("probabilities").at(option.key).get<double>());
                }
                rows.push_back(row);
            } else {
                rows.push_back(nullptr);
            }
        }
        out["probabilities"] = rows;
        // when each layer's output was ready, ms after the forward pass started (first variant)
        json ms = json::array();
        for (const double v : layer_ms_results[first_task]) {
            if (v < 0) ms.push_back(nullptr); else ms.push_back(v);
        }
        out["ms"] = ms;
        json timings = json::object();
        timings["forward_ms"] = lens_process_ms;
        timings["readout_ms"] = lens_readout_ms;
        out["timings"] = timings;
        const auto & tk = token_results[first_task];
        if (!tk.pieces.empty()) {
            json pieces = json::array();
            for (const auto & p : tk.pieces) pieces.push_back(p);
            out["tokens"] = pieces;
            json change = json::array();
            for (const auto & layer : tk.change) {
                json row = json::array();
                for (const float v : layer) row.push_back((double) v);
                change.push_back(row);
            }
            out["token_change"] = change;
            if (!tk.attention.empty()) {
                json attn = json::object();
                for (const auto & [il, heads] : tk.attention) {
                    json hs = json::array();
                    for (const auto & head : heads) {
                        json r = json::array();
                        for (const float v : head) r.push_back((double) v);
                        hs.push_back(r);
                    }
                    attn[std::to_string(il)] = hs;
                }
                out["attention"] = attn;
            }
        }
        return out;
    }

    // called with the mutex held
    std::string answer(const std::string & request) {
        if (decision.type == COMMON_DECISION_TYPE_NONE) {
            return error_json("This model is not a decision model", "not_supported_error");
        }
        json body;
        try {
            body = json::parse(request);
        } catch (const std::exception & e) {
            return error_json(std::string("invalid JSON: ") + e.what(), "invalid_request_error");
        }
        try {
            const auto questions = decision.parse_questions(body);
            std::vector<raw_buffer> files;
            const json state = decision.prepare_state(decision.parse_state(body, files));
            if (!files.empty() && !mctx) {
                return error_json("this model was loaded without a projector (mmproj), it cannot read images", "not_supported_error");
            }
            if (!files.empty() && !decision.can_use_images()) {
                return error_json("this decision model does not read images", "not_supported_error");
            }
            const mtmd_helper_init_opt media_opt = mtmd_helper_init_opt_default();

            std::vector<server_task> tasks;
            if (decision.is_joint()) {
                server_task task(SERVER_TASK_TYPE_DECISION);
                decision.fill_task_joint(state, questions, files, mctx, media_opt, task);
                tasks.push_back(std::move(task));
            } else {
                for (const auto & question : questions) {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        server_task task(SERVER_TASK_TYPE_DECISION);
                        decision.fill_task(state, questions, question, variant, files, mctx, media_opt, task);
                        tasks.push_back(std::move(task));
                    }
                }
            }

            // "layers": true (an extension of this library) adds the answer of every layer, a logit lens
            const bool want_layers = json_value(body, "layers", false);
            std::vector<std::vector<float>> results;
            std::vector<std::vector<std::vector<float>>> layer_results; // [task][layer][option]
            std::vector<std::vector<double>> layer_ms_results;            // [task][layer]
            std::vector<lens_tokens_result> token_results;                 // [task]
            int32_t n_tokens = 0;
            std::string layers_error;
            for (const auto & task : tasks) {
                std::vector<std::vector<float>> layers;
                std::vector<double> ms;
                lens_tokens_result per_token;
                const auto & td = task.decision;
                const bool lens_possible = !decision.is_joint() && !td.pooled && td.n_scores == 0 &&
                    (!td.labels.empty() || !td.markers.empty());
                if (want_layers && lens_possible) {
                    try {
                        results.push_back(run(task, &layers, &ms, &per_token));
                        if (layers.empty() && model->hparams.n_layer_decision > 0 && layers_error.empty()) {
                            layers_error = "this model answers through a trained decision head this library cannot re-run yet";
                        }
                    } catch (const std::exception & e) {
                        layers_error = e.what();
                        results.push_back(run(task));
                    }
                } else {
                    results.push_back(run(task));
                    if (want_layers && layers_error.empty()) {
                        layers_error = "layer readout is not available for models read from a pooled embedding";
                    }
                }
                layer_results.push_back(std::move(layers));
                layer_ms_results.push_back(std::move(ms));
                token_results.push_back(std::move(per_token));
                n_tokens += task.n_tokens();
            }

            json answers = json::object();
            size_t i_result = 0;
            size_t i_score  = 0;
            for (const auto & question : questions) {
                std::vector<std::vector<float>> scores;
                if (decision.is_joint()) {
                    const auto & all = results[0];
                    if (i_score + question.options.size() > all.size()) {
                        throw std::runtime_error("the joint head returned fewer scores than options");
                    }
                    scores.emplace_back(all.begin() + i_score, all.begin() + i_score + question.options.size());
                    i_score += question.options.size();
                } else {
                    for (size_t variant = 0; variant < decision.n_variants(question); variant++) {
                        scores.push_back(results[i_result++]);
                    }
                }
                json answer = decision.format_answer(question, scores, !files.empty());
                if (want_layers) {
                    answer["layers"] = layers_answer(question, scores.size(), i_result - scores.size(), layer_results, layer_ms_results, token_results, layers_error, !files.empty());
                }
                answers[question.id] = answer;
            }

            json usage = json::object();
            usage["input_tokens"]  = n_tokens;
            if (!files.empty()) {
                usage["image_encode_ms"] = media_encode_ms;
            }
            usage["output_tokens"] = 0;
            json out = json::object();
            out["model"]   = model_name;
            out["answers"] = answers;
            out["usage"]   = usage;
            return out.dump_safe();
        } catch (const std::invalid_argument & e) {
            return error_json(e.what(), "invalid_request_error");
        } catch (const std::exception & e) {
            return error_json(e.what(), "server_error");
        }
    }
};

extern "C" {

systemone_params systemone_default_params(void) {
    systemone_params p;
    p.n_gpu_layers = -1;
    p.n_ctx        = 4096;
    p.n_threads    = 0;
    p.verbose      = false;
    p.repack       = true;
    p.flash_attn   = true;
    p.mmproj_path  = nullptr;
    return p;
}

systemone * systemone_load(const char * model_path, const systemone_params * params_in) {
    systemone_params params = params_in ? *params_in : systemone_default_params();
    if (!params.verbose) {
        llama_log_set(quiet_log, nullptr);
    }
    static std::once_flag backend_once;
    std::call_once(backend_once, [] { llama_backend_init(); });

    auto so = std::make_unique<systemone>();
    so->n_ctx = params.n_ctx ? params.n_ctx : 4096;

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = params.n_gpu_layers < 0 ? 999 : params.n_gpu_layers;
    mparams.use_extra_bufts = params.repack; // repacked copies of the weights: ~1 GB more RAM for d1-2B, faster on ARM
    so->model = llama_model_load_from_file(model_path, mparams);
    if (!so->model) {
        fprintf(stderr, "systemone: failed to load %s\n", model_path);
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = so->n_ctx;
    cparams.n_batch   = so->n_ctx; // the whole prompt goes through in one batch
    cparams.n_ubatch  = so->n_ctx;
    cparams.n_seq_max = 1;
    cparams.cb_eval           = systemone::cb_eval; // only reads when a request asks for layers
    cparams.cb_eval_user_data = so.get();
    cparams.no_perf   = true;
    cparams.flash_attn_type = params.flash_attn ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    so->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(so->n_threads);
        pin_to_fast_cores(tpp, so->n_threads);
        so->threadpool = ggml_threadpool_new(&tpp);
    }
    if (params.n_threads > 0) {
        cparams.n_threads       = params.n_threads;
        cparams.n_threads_batch = params.n_threads;
    }
    so->ctx = llama_init_from_model(so->model, cparams);
    if (!so->ctx) {
        fprintf(stderr, "systemone: failed to create the context\n");
        return nullptr;
    }
    if (so->threadpool) {
        llama_attach_threadpool(so->ctx, so->threadpool, so->threadpool);
    }

    if (params.mmproj_path && params.mmproj_path[0] != '\0') {
        if (!params.verbose) {
            mtmd_helper_log_set(quiet_log, nullptr);
        }
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu         = params.n_gpu_layers != 0;
        mp.print_timings   = false;
        mp.n_threads       = so->n_threads;
        mp.media_marker    = get_media_marker();
        mp.flash_attn_type = cparams.flash_attn_type;
        mp.warmup          = false;
        // the image shares one batch with the text: leave the text half of it
        const int n_max = (int) so->n_ctx / 2;
        if (mp.image_max_tokens <= 0 || mp.image_max_tokens > n_max) {
            mp.image_max_tokens = n_max;
            if (mp.image_min_tokens > n_max) mp.image_min_tokens = n_max;
        }
        so->mctx = mtmd_init_from_file(params.mmproj_path, so->model, mp);
        if (!so->mctx) {
            fprintf(stderr, "systemone: failed to load the projector %s\n", params.mmproj_path);
            return nullptr;
        }
    }

    try {
        so->decision.init(so->model);
    } catch (const std::exception & e) {
        fprintf(stderr, "systemone: failed to init the decision model: %s\n", e.what());
        return nullptr;
    }
    so->type_name = decision_type_name(so->decision.type);

    const char * slash = strrchr(model_path, '/');
    so->model_name = slash ? slash + 1 : model_path;
    return so.release();
}

void systemone_free(systemone * so) {
    delete so;
}

const char * systemone_model_type(const systemone * so) {
    return so ? so->type_name.c_str() : "none";
}

char * systemone_decide(systemone * so, const char * request_json) {
    // nothing may cross the C ABI: an exception escaping here would abort the app
    static const char fallback[] = "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}";
    std::string out;
    try {
        if (!so || !request_json) {
            out = error_json("no model or no request", "invalid_request_error");
        } else {
            out = so->decide(request_json);
        }
    } catch (...) {
        out.clear();
    }
    const char * src = out.empty() ? fallback : out.c_str();
    const size_t n   = out.empty() ? sizeof(fallback) : out.size() + 1;
    char * s = (char *) malloc(n);
    if (s) {
        memcpy(s, src, n);
    }
    return s;
}

char * systemone_model_info(const systemone * so) {
    std::string out;
    try {
        out = so ? so->model_info().dump_safe() : error_json("no model", "invalid_request_error");
    } catch (...) {
        out = "{\"error\":{\"message\":\"internal error\",\"type\":\"server_error\"}}";
    }
    char * s = (char *) malloc(out.size() + 1);
    if (s) {
        memcpy(s, out.c_str(), out.size() + 1);
    }
    return s;
}

void systemone_free_string(char * s) {
    free(s);
}

const char * systemone_last_error(const systemone * so) {
    return so ? so->last_error.c_str() : "";
}

} // extern "C"
