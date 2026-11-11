// systemone-cli: one JSON request per line on stdin, one JSON answer per line on stdout.
//
//   echo '{"state": "...", "questions": {...}}' | systemone-cli -m d1-2B-Q4_K_M.gguf
//
// The same body llama-server's /v1/systemone takes, so the two can be compared (tests/parity.py).

#include "systemone.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

static void usage(const char * argv0) {
    fprintf(stderr,
            "usage: %s -m MODEL.gguf [-ngl N] [-c N_CTX] [-t THREADS] [--no-repack] [--time] [-v]\n"
            "  reads one JSON request per line from stdin, writes one JSON answer per line\n"
            "  --info       print what the model is made of (JSON) and exit\n"
            "  --time       adds \"_wall_ms\" to every answer\n"
            "  --no-repack  no repacked copy of the weights: about half the RAM, slower on ARM\n"
            "  --mmproj F   the projector of a multimodal model; requests can then carry \"images\": [data URLs]\n"
            "  --no-flash-attn  plain attention, so \"layers\": true can return the attention maps\n",
            argv0);
}

int main(int argc, char ** argv) {
    const char * model = nullptr;
    systemone_params params = systemone_default_params();
    bool info = false;
    bool time = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value\n", name);
                exit(2);
            }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") {
            model = next("-m");
        } else if (a == "-ngl" || a == "--n-gpu-layers") {
            params.n_gpu_layers = atoi(next("-ngl"));
        } else if (a == "-c" || a == "--ctx-size") {
            params.n_ctx = (uint32_t) atoi(next("-c"));
        } else if (a == "-t" || a == "--threads") {
            params.n_threads = atoi(next("-t"));
        } else if (a == "--info") {
            info = true;
        } else if (a == "--mmproj") {
            params.mmproj_path = next("--mmproj");
        } else if (a == "--no-flash-attn") {
            params.flash_attn = false;
        } else if (a == "--no-repack") {
            params.repack = false;
        } else if (a == "--time") {
            time = true;
        } else if (a == "-v" || a == "--verbose") {
            params.verbose = true;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }
    if (!model) {
        usage(argv[0]);
        return 2;
    }

    systemone * so = systemone_load(model, &params);
    if (!so) {
        return 1;
    }
    fprintf(stderr, "systemone-cli: %s loaded, decision model type %s\n", model, systemone_model_type(so));
    if (info) {
        char * text = systemone_model_info(so);
        printf("%s\n", text ? text : "{}");
        systemone_free_string(text);
        systemone_free(so);
        return 0;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        char * answer = systemone_decide(so, line.c_str());
        if (!answer) {
            fprintf(stderr, "systemone-cli: out of memory\n");
            systemone_free(so);
            return 1;
        }
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::string out = answer;
        systemone_free_string(answer);
        if (time && !out.empty() && out.back() == '}') {
            out.pop_back();
            out += ", \"_wall_ms\": " + std::to_string(ms) + "}";
        }
        std::cout << out << std::endl;
    }
    systemone_free(so);
    return 0;
}
