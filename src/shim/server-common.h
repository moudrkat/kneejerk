// What server-decision.cpp needs from llama.cpp's tools/server/server-common.h, and nothing else.
// The upstream file is kept as is (src/upstream/), so it can be diffed against new llama.cpp releases;
// this header stands in for the server it was written in.
#pragma once

#include "common.h"
#include "chat.h"
#include "llama.h"
#include "json.h"
#include "base64.hpp"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <map>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;
using raw_buffer = std::vector<uint8_t>;

#define SRV_INF(fmt, ...) fprintf(stderr, "systemone: " fmt, __VA_ARGS__)

template <typename T>
static T json_value(const json & body, const std::string & key, const T & default_value) {
    if (body.contains(key) && !body.at(key).is_null()) {
        try {
            return body.at(key).get<T>();
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

// The prompt of a task: text tokens, and media chunks (an image, an audio clip) that occupy several
// positions each, held as LLAMA_TOKEN_NULL in the token list. Same layout as the server's server_tokens.
struct server_tokens {
    server_tokens() = default;
    server_tokens(const llama_tokens & tokens, bool has_mtmd) : has_mtmd(has_mtmd), tokens(tokens) {}
    server_tokens(mtmd::input_chunks & mtmd_chunks, bool has_mtmd) : has_mtmd(has_mtmd) {
        for (size_t i = 0; i < mtmd_chunks.size(); ++i) {
            push_back(mtmd_chunks[i]);
        }
    }

    server_tokens(const server_tokens &) = delete;
    server_tokens & operator=(const server_tokens &) = delete;
    server_tokens(server_tokens &&) = default;
    server_tokens & operator=(server_tokens &&) = default;

    llama_token operator[](size_t index) const { return tokens[index]; }
    size_t size() const { return tokens.size(); }
    llama_tokens get_text_tokens() const { return tokens; }

    void push_back(llama_token tok) { tokens.push_back(tok); }

    void push_back(const mtmd_input_chunk * chunk) {
        const auto type = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            const size_t n_tokens  = mtmd_input_chunk_get_n_tokens(chunk);
            const size_t start_idx = tokens.size();
            tokens.insert(tokens.end(), n_tokens, LLAMA_TOKEN_NULL);
            map_idx_to_media[start_idx] = mtmd::input_chunk_ptr(mtmd_input_chunk_copy(chunk));
            has_mtmd = true;
        } else if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n_tokens = 0;
            const auto * text_tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            tokens.insert(tokens.end(), text_tokens, text_tokens + n_tokens);
        } else {
            throw std::runtime_error("invalid media chunk type");
        }
    }

    size_t get_common_prefix(const server_tokens & b) const {
        const size_t n = std::min(tokens.size(), b.tokens.size());
        size_t i = 0;
        while (i < n && tokens[i] == b.tokens[i] && tokens[i] != LLAMA_TOKEN_NULL) {
            i++;
        }
        return i;
    }

    const mtmd::input_chunk_ptr & find_chunk(size_t idx) const {
        const auto it = map_idx_to_media.find(idx);
        if (it == map_idx_to_media.end()) {
            throw std::runtime_error("Chunk not found");
        }
        return it->second;
    }

    bool has_mtmd = false;
    llama_tokens tokens;
    std::map<size_t, mtmd::input_chunk_ptr> map_idx_to_media;
};

// The marker a prompt carries where a media file goes. The mtmd context is created with the same one.
inline const char * get_media_marker() {
    return "<__media__>";
}

// standard base64, padding optional, whitespace skipped; throws on any other character
inline raw_buffer decode_base64(const std::string & in) {
    raw_buffer out;
    out.reserve(in.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    for (const unsigned char c : in) {
        int v;
        if (c >= 'A' && c <= 'Z') v = c - 'A';
        else if (c >= 'a' && c <= 'z') v = c - 'a' + 26;
        else if (c >= '0' && c <= '9') v = c - '0' + 52;
        else if (c == '+' || c == '-') v = 62;
        else if (c == '/' || c == '_') v = 63;
        else if (c == '=' ) break;
        else if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        else throw std::invalid_argument("invalid base64 data");
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t) ((acc >> bits) & 0xFF));
        }
    }
    return out;
}

// Media arrive as base64 data URLs only: no downloads, no local paths from a request.
inline void handle_media(std::vector<raw_buffer> & out_files, const std::string & url, const std::string & /*media_path*/) {
    if (url.rfind("data:", 0) != 0) {
        throw std::invalid_argument("media must be base64 data URLs (data:image/...;base64,...)");
    }
    const size_t comma = url.find(',');
    if (comma == std::string::npos || url.find(',', comma + 1) != std::string::npos) {
        throw std::invalid_argument("Invalid uri-encoded base64 value");
    }
    const std::string head = url.substr(0, comma);
    if (head.rfind("data:image/", 0) != 0 && head.rfind("data:audio/", 0) != 0) {
        throw std::invalid_argument("Invalid uri format: " + head);
    }
    if (head.size() < 6 || head.compare(head.size() - 6, 6, "base64") != 0) {
        throw std::invalid_argument("uri must be base64 encoded");
    }
    out_files.push_back(decode_base64(url.substr(comma + 1)));
}

// The prompt with its media: the files become bitmaps, mtmd splits the prompt at the markers.
// Same as the server's process_mtmd_prompt(); throws if the model was loaded without a projector.
inline server_tokens process_mtmd_prompt(mtmd_context * mctx, const std::string & prompt,
                                         const std::vector<raw_buffer> & files, const mtmd_helper_init_opt & opt) {
    if (!mctx) {
        throw std::invalid_argument("this model was loaded without a projector (mmproj), it cannot read images");
    }
    mtmd::bitmaps bitmaps;
    for (const auto & file : files) {
        auto out = mtmd_helper_bitmap_init_from_buf(mctx, file.data(), file.size(), false, opt);
        if (!out.bitmap) {
            throw std::invalid_argument("Failed to load image or audio file");
        }
        {
            // the id is a hash of the file: the same picture asked twice is encoded once (see systemone::run)
            uint64_t h = 1469598103934665603ULL;
            for (const uint8_t b : file) { h ^= b; h *= 1099511628211ULL; }
            char id[32];
            snprintf(id, sizeof(id), "%016llx", (unsigned long long) h);
            mtmd_bitmap_set_id(out.bitmap, id);
        }
        bitmaps.entries.emplace_back(out.bitmap);
        if (out.video_ctx) {
            mtmd_helper_video_free(out.video_ctx);
            throw std::invalid_argument("video is not supported");
        }
    }
    mtmd_input_text inp_txt = { prompt.data(), prompt.size(), /* add_special */ true, /* parse_special */ true };
    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = bitmaps.c_ptr();
    if (mtmd_tokenize(mctx, chunks.ptr.get(), &inp_txt, bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
        throw std::runtime_error("Failed to tokenize prompt");
    }
    return server_tokens(chunks, true);
}

// What a model of this decision type can output; copied from server-common.cpp.
inline std::vector<std::string> server_model_output_modalities(common_decision_type decision_type, bool has_cap_chat, bool has_cap_tts) {
    switch (decision_type) {
        case COMMON_DECISION_TYPE_OPENJEV:
        case COMMON_DECISION_TYPE_LEV:
        case COMMON_DECISION_TYPE_KEV:
        case COMMON_DECISION_TYPE_NIMBLE:
        case COMMON_DECISION_TYPE_LAYA:
        case COMMON_DECISION_TYPE_CLEF:
        case COMMON_DECISION_TYPE_PPLX_DECIDER:
        case COMMON_DECISION_TYPE_VLLM_DECISION2:
        case COMMON_DECISION_TYPE_VLLM_D3:
        case COMMON_DECISION_TYPE_LFM2_D1:
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
        case COMMON_DECISION_TYPE_H2O_LIGHTNING:
        case COMMON_DECISION_TYPE_MINICPM_OCR_JEV:
        case COMMON_DECISION_TYPE_LFM3_D1:
        case COMMON_DECISION_TYPE_LFM3_D1_OMNI:
            return {"decisions"};
        case COMMON_DECISION_TYPE_GRANITE_SWITCH:
            return {"text", "decisions"};
        default:
            if (has_cap_chat || !has_cap_tts) {
                return {"text"};
            }
            return {"audio"};
    }
}
