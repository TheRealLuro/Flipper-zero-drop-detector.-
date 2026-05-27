// Unified Flipper Drop Detector workbench.
// Dear ImGui + Win32 + DirectX 11. One .exe that does dataset build, train,
// predict, and fine-tune -- replacing data.py, main.cpp's CLI, tune_gui.py.

#define NOMINMAX  // Stop windows.h from #defining min/max macros.
#include "bean_tensor.hpp"

#include "imgui.h"
#include "backends/imgui_impl_win32.h"
#include "backends/imgui_impl_dx11.h"
#include <d3d11.h>
#include <tchar.h>
#include <windows.h>
#include <commdlg.h>
#include <shobjidl.h>
#include <objbase.h>
#include <dwmapi.h>
#include <windowsx.h>

#include <atomic>
#include <cstdarg>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// Font handles, populated in WinMain.
static ImFont* g_font_ui   = nullptr;
static ImFont* g_font_h1   = nullptr;
static ImFont* g_font_h2   = nullptr;
static ImFont* g_font_mono = nullptr;
static float   g_dpi_scale = 1.0f;

// Custom title bar layout constants (in unscaled pixels; multiply by g_dpi_scale at use).
constexpr float TITLEBAR_H_UNSCALED   = 36.0f;
constexpr float WIN_BTN_W_UNSCALED    = 46.0f;
constexpr int   NUM_WIN_BTNS          = 3;

// Narrow-layout breakpoint (DPI-scaled). Below this content width, tabs go single-column.
constexpr float NARROW_PX_UNSCALED    = 900.0f;

enum class Tab : int { Build = 0, Train, Predict, Tune, Runs };
static constexpr int NUM_TABS = 5;
static const char* tab_label(Tab t) {
    switch (t) {
        case Tab::Build:   return "Build dataset";
        case Tab::Train:   return "Train";
        case Tab::Predict: return "Predict";
        case Tab::Tune:    return "Tune";
        case Tab::Runs:    return "Runs";
    }
    return "?";
}

// ============================================================================
// Logger -- thread-safe append-only string the UI snapshots every frame.
// ============================================================================
class Logger {
    std::mutex mu_;
    std::string buf_;
public:
    void clear() { std::lock_guard<std::mutex> lk(mu_); buf_.clear(); }
    void log(const char* s) {
        std::lock_guard<std::mutex> lk(mu_);
        buf_ += s;
        if (buf_.size() > 400000) buf_.erase(0, buf_.size() - 400000);
    }
    void logf(const char* fmt, ...) {
        char tmp[4096];
        va_list ap; va_start(ap, fmt);
        std::vsnprintf(tmp, sizeof(tmp), fmt, ap);
        va_end(ap);
        log(tmp);
    }
    void snapshot(std::string& dst) { std::lock_guard<std::mutex> lk(mu_); dst = buf_; }
    size_t size() { std::lock_guard<std::mutex> lk(mu_); return buf_.size(); }
};

// ============================================================================
// AppState -- all UI fields + worker thread plumbing.
// ============================================================================
struct AppState {
    std::atomic<bool> busy{false};
    std::atomic<bool> dirty_log{true};
    Logger log;
    std::string log_snapshot;

    Tab active_tab = Tab::Build;

    // Build dataset tab
    char build_src[260]   = "data\\drop_detect";
    char build_train[260] = "data\\TRAIN.bin";
    char build_test[260]  = "data\\TEST.bin";
    int  build_test_pct   = 20;
    int  build_seed       = 42;

    // Train tab
    char train_train_bin[260] = "data\\TRAIN.bin";
    char train_test_bin[260]  = "data\\TEST.bin";
    char train_model_out[260] = "data\\model.json";
    float train_lr            = 0.01f;
    int   train_epochs        = 15;
    char train_note[128]      = "";   // optional label saved with the archived run

    // Model architecture (used by train; persisted to model.json so predict/tune
    // can rebuild the same shape automatically).
    int arch_conv_channels = 8;   // Conv1D output channels
    int arch_conv_kernel   = 7;   // Conv1D kernel size
    int arch_conv_stride   = 2;   // Conv1D stride
    int arch_hidden_layers = 0;   // # of hidden Dense layers between pool and output (0..3)
    int arch_hidden_units  = 16;  // units per hidden Dense layer

    // Predict tab
    char predict_csv[1024]      = "";
    char predict_model[260]     = "data\\model.json";
    char predict_train_bin[260] = "data\\TRAIN.bin";

    // Tune tab
    char tune_user_dir[260]   = "data\\user_data";
    char tune_model_in[260]   = "data\\model.json";
    char tune_train_bin[260]  = "data\\TRAIN.bin";
    char tune_test_bin[260]   = "data\\TEST.bin";
    float tune_lr             = 0.005f;
    int   tune_epochs         = 5;
    bool  tune_keep_old       = false;

    // Runs tab
    char runs_dir[260]        = "runs";
    int  runs_selected        = -1;            // currently selected row in the runs table
    bool runs_refresh         = true;          // request re-scan on next render
    float runs_last_scan      = -1e9f;         // throttle filesystem scans

    // Per-card collapse state (id -> collapsed). Persists for the session.
    std::unordered_map<std::string, bool> card_collapsed;
};

// Filled in by train_loop so worker_train can archive these to a run folder.
struct TrainMetrics {
    double final_loss     = 0.0;
    int    final_correct  = 0;
    int    final_total    = 0;
    int    train_cc[4]    = {};
    int    train_ct[4]    = {};
    int    eval_correct   = 0;
    int    eval_total     = 0;
    int    eval_cc[4]     = {};
    int    eval_ct[4]     = {};
};

// A single archived training run loaded from runs/<timestamp>/metrics.json.
struct RunInfo {
    std::filesystem::path folder;
    std::string timestamp;
    std::string note;
    ArchSpec    arch;
    long long   params       = 0;
    int         epochs       = 0;
    double      learning_rate = 0;
    double      final_loss   = 0;
    double      final_train_acc = 0;
    int         eval_correct = 0;
    int         eval_total   = 0;
    double      test_accuracy = 0;
    double      per_class_acc[4] = {};  // idle, walking, fidget, drop
};

static void archive_run(AppState& s, const ArchSpec& arch, const TrainMetrics& m);
static std::vector<RunInfo> scan_runs(const std::filesystem::path& dir);
static void render_runs_tab(AppState& s);

// Standard "must own the busy flag" launcher.
static void launch_worker(AppState& s, std::function<void()> fn) {
    if (s.busy.exchange(true)) return;
    std::thread([fn = std::move(fn), &s]() {
        try { fn(); }
        catch (const std::exception& e) { s.log.logf("\nERROR: %s\n", e.what()); }
        catch (...) { s.log.log("\nERROR: unknown exception\n"); }
        s.busy.store(false);
    }).detach();
}

// ============================================================================
// Dataset build  (port of data.py)
// ============================================================================
namespace dataset_build {

static const char* CLASS_NAMES[4] = { "idle", "walking", "fidget", "drop" };
constexpr int WINDOW = 250;
constexpr int STRIDE = 125;
constexpr int FEATURES = 4;

struct FilesByClass {
    std::vector<std::pair<fs::path, int>> entries; // (csv path, label)
};

static std::map<int, std::vector<fs::path>> enumerate(const fs::path& base) {
    std::map<int, std::vector<fs::path>> out;
    for (int lbl = 0; lbl < 4; lbl++) {
        fs::path sub = base / CLASS_NAMES[lbl];
        if (!fs::is_directory(sub)) continue;
        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(sub)) {
            if (e.is_regular_file() && e.path().extension() == ".csv") files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        if (!files.empty()) out[lbl] = std::move(files);
    }
    return out;
}

static void stratified_split(const std::map<int, std::vector<fs::path>>& by_class,
                             double test_ratio, uint32_t seed,
                             std::vector<std::pair<fs::path,int>>& train_out,
                             std::vector<std::pair<fs::path,int>>& test_out) {
    std::mt19937 rng(seed);
    for (auto& [lbl, files] : by_class) {
        std::vector<size_t> idx(files.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        size_t n_test = files.size() > 1 ? std::max<size_t>(1, std::lround(files.size() * test_ratio)) : 0;
        size_t n_train = files.size() - n_test;
        for (size_t i = 0; i < n_train; i++)         train_out.emplace_back(files[idx[i]], lbl);
        for (size_t i = n_train; i < files.size(); i++) test_out.emplace_back(files[idx[i]], lbl);
    }
}

// Build raw windowed feature tensor (N, T=249, F=4) from a list of (csv, label) pairs.
// Output X is flat row-major.
static void build_from_files(const std::vector<std::pair<fs::path,int>>& files,
                             std::vector<float>& X, std::vector<int32_t>& y,
                             int32_t& N_out, int32_t T = WINDOW - 1, int32_t F = FEATURES) {
    X.clear(); y.clear();
    for (auto& [path, lbl] : files) {
        auto rows = read_imu_csv(path);
        if (static_cast<int>(rows.size()) < WINDOW) continue;
        auto sq = [](float a, float b, float c) { return a*a + b*b + c*c; };
        for (size_t start = 0; start + WINDOW <= rows.size(); start += STRIDE) {
            // Append T*F floats for this window
            for (int t = 1; t < WINDOW; t++) {
                const auto& cur  = rows[start + t];
                const auto& prev = rows[start + t - 1];
                float a      = sq(cur[0],  cur[1],  cur[2]);
                float g      = sq(cur[3],  cur[4],  cur[5]);
                float a_prev = sq(prev[0], prev[1], prev[2]);
                float g_prev = sq(prev[3], prev[4], prev[5]);
                X.push_back(a);
                X.push_back(g);
                X.push_back(a - a_prev);
                X.push_back(g - g_prev);
            }
            y.push_back(lbl);
        }
    }
    N_out = static_cast<int32_t>(y.size());
}

static void fit_normalization(const std::vector<float>& X, int32_t N, int32_t T, int32_t F,
                              std::vector<float>& mean, std::vector<float>& std_) {
    mean.assign(F, 0.0f);
    std_.assign(F, 0.0f);
    const size_t total = static_cast<size_t>(N) * T;
    if (total == 0) { for (int f = 0; f < F; f++) std_[f] = 1.0f; return; }
    // Mean
    for (size_t i = 0; i < total; i++)
        for (int f = 0; f < F; f++) mean[f] += X[i * F + f];
    for (int f = 0; f < F; f++) mean[f] = static_cast<float>(mean[f] / total);
    // Std
    for (size_t i = 0; i < total; i++)
        for (int f = 0; f < F; f++) {
            double d = X[i * F + f] - mean[f];
            std_[f] += static_cast<float>(d * d);
        }
    for (int f = 0; f < F; f++) {
        std_[f] = std::sqrt(static_cast<double>(std_[f]) / total);
        if (std_[f] < 1e-6f) std_[f] = 1.0f;
    }
}

static void apply_normalization(std::vector<float>& X, int32_t N, int32_t T, int32_t F,
                                const std::vector<float>& mean, const std::vector<float>& std_) {
    const size_t total = static_cast<size_t>(N) * T;
    for (size_t i = 0; i < total; i++)
        for (int f = 0; f < F; f++)
            X[i * F + f] = (X[i * F + f] - mean[f]) / std_[f];
}

static void save_bin(const fs::path& path,
                     const std::vector<float>& X, const std::vector<int32_t>& y,
                     const std::vector<float>& mean, const std::vector<float>& std_,
                     int32_t N, int32_t T, int32_t F) {
    FILE* f = fopen(path.string().c_str(), "wb");
    if (!f) throw std::runtime_error("Could not open " + path.string() + " for write");
    std::fwrite(&N, sizeof(int32_t), 1, f);
    std::fwrite(&T, sizeof(int32_t), 1, f);
    std::fwrite(&F, sizeof(int32_t), 1, f);
    std::fwrite(mean.data(), sizeof(float), mean.size(), f);
    std::fwrite(std_.data(), sizeof(float), std_.size(), f);
    std::fwrite(X.data(), sizeof(float), X.size(), f);
    std::fwrite(y.data(), sizeof(int32_t), y.size(), f);
    std::fclose(f);
}

} // namespace dataset_build

// ============================================================================
// Workers (one per tab)
// ============================================================================

// --- Build dataset worker ---
static void worker_build_dataset(AppState& s) {
    namespace db = dataset_build;
    Logger& L = s.log;
    L.logf("=== Build dataset ===\n");
    fs::path src(s.build_src);
    if (!fs::is_directory(src)) { L.logf("Source folder not found: %s\n", src.string().c_str()); return; }

    auto by_class = db::enumerate(src);
    if (by_class.empty()) { L.log("No class folders found.\n"); return; }
    for (auto& [lbl, files] : by_class)
        L.logf("  %s/  %zu files\n", db::CLASS_NAMES[lbl], files.size());

    std::vector<std::pair<fs::path,int>> train_files, test_files;
    db::stratified_split(by_class, s.build_test_pct / 100.0, s.build_seed, train_files, test_files);
    L.logf("Split: %zu train, %zu test (test_ratio=%d%%, seed=%d)\n",
           train_files.size(), test_files.size(), s.build_test_pct, s.build_seed);

    std::vector<float>   X_train, X_test;
    std::vector<int32_t> y_train, y_test;
    int32_t N_train, N_test;
    constexpr int32_t T = db::WINDOW - 1, F = db::FEATURES;

    L.log("Extracting train windows...\n");
    db::build_from_files(train_files, X_train, y_train, N_train);
    L.logf("  X_train: (%d, %d, %d)\n", N_train, T, F);

    L.log("Extracting test windows...\n");
    db::build_from_files(test_files, X_test, y_test, N_test);
    L.logf("  X_test:  (%d, %d, %d)\n", N_test, T, F);

    int counts_tr[4] = {}, counts_te[4] = {};
    for (auto v : y_train) if (v >= 0 && v < 4) counts_tr[v]++;
    for (auto v : y_test)  if (v >= 0 && v < 4) counts_te[v]++;
    L.logf("  train counts: idle=%d walking=%d fidget=%d drop=%d\n",
           counts_tr[0], counts_tr[1], counts_tr[2], counts_tr[3]);
    L.logf("  test counts:  idle=%d walking=%d fidget=%d drop=%d\n",
           counts_te[0], counts_te[1], counts_te[2], counts_te[3]);

    if (N_train == 0) { L.log("ERROR: train set is empty.\n"); return; }

    std::vector<float> mean, std_;
    db::fit_normalization(X_train, N_train, T, F, mean, std_);
    L.logf("Normalization (from train only): mean=[%.4g, %.4g, %.4g, %.4g]  std=[%.4g, %.4g, %.4g, %.4g]\n",
           mean[0], mean[1], mean[2], mean[3], std_[0], std_[1], std_[2], std_[3]);

    db::apply_normalization(X_train, N_train, T, F, mean, std_);
    if (N_test > 0) db::apply_normalization(X_test, N_test, T, F, mean, std_);

    db::save_bin(s.build_train, X_train, y_train, mean, std_, N_train, T, F);
    db::save_bin(s.build_test,  X_test,  y_test,  mean, std_, N_test,  T, F);
    L.logf("Wrote %s and %s.\n", s.build_train, s.build_test);

    L.log("=== done ===\n");
}

// --- Train worker (one model end-to-end, with fused softmax+CE gradient) ---
static void train_loop(AppState& s, Layers::Sequential& model,
                       const std::vector<float>& X, const std::vector<int32_t>& y,
                       int N, int T, int F, int epochs, float lr,
                       const std::vector<float>* X_eval, const std::vector<int32_t>* y_eval,
                       int N_eval,
                       TrainMetrics* metrics_out = nullptr) {
    Logger& L = s.log;
    constexpr int NUM_CLASSES = 4;
    const char* NAMES[4] = {"idle","walking","fidget","drop"};
    std::mt19937 rng(42);
    std::vector<size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    for (int epoch = 0; epoch < epochs; epoch++) {
        std::ranges::shuffle(idx, rng);
        double total_loss = 0.0;
        int correct = 0;
        int cc[4] = {}, ct[4] = {};
        for (size_t si = 0; si < idx.size(); si++) {
            const size_t s_i = idx[si];
            Tensors::Tensor input({static_cast<size_t>(F), static_cast<size_t>(T)}, FP32);
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++)
                    input.set({static_cast<size_t>(f), static_cast<size_t>(t)},
                              X[s_i * T * F + t * F + f]);

            Tensors::Tensor target({static_cast<size_t>(NUM_CLASSES)}, FP32);
            target.all_zeros();
            target.set({static_cast<size_t>(y[s_i])}, 1.0);

            auto output = model.forward(input);
            double loss = 0.0;
            for (size_t i = 0; i < output.numel(); i++)
                loss -= target.at({i}) * std::log(output.at({i}) + 1e-9);
            total_loss += loss;

            int pred = 0;
            for (int c = 1; c < NUM_CLASSES; c++)
                if (output.at({(size_t)c}) > output.at({(size_t)pred})) pred = c;
            ct[y[s_i]]++;
            if (pred == y[s_i]) { correct++; cc[y[s_i]]++; }

            Tensors::Tensor grad({static_cast<size_t>(NUM_CLASSES)}, FP32);
            for (size_t i = 0; i < grad.numel(); i++)
                grad.set({i}, output.at({i}) - target.at({i}));
            Tensors::Tensor g = grad;
            for (int li = (int)model._layers.size() - 2; li >= 0; li--)
                g = model._layers[li]->backward(g, lr);
        }
        double avg = total_loss / N;
        if (!std::isfinite(avg)) { L.logf("Loss became non-finite at epoch %d -- aborting.\n", epoch + 1); break; }
        L.logf("Epoch %2d/%d  loss=%.4f  acc=%5.1f%%  [", epoch + 1, epochs, avg,
               100.0 * correct / N);
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (c) L.log(" ");
            double pct = ct[c] ? 100.0 * cc[c] / ct[c] : 0.0;
            L.logf("%s=%.0f%%", NAMES[c], pct);
        }
        L.log("]\n");

        // Capture final-epoch metrics so worker_train can archive them.
        if (metrics_out) {
            metrics_out->final_loss    = avg;
            metrics_out->final_correct = correct;
            metrics_out->final_total   = N;
            for (int c = 0; c < NUM_CLASSES; c++) {
                metrics_out->train_cc[c] = cc[c];
                metrics_out->train_ct[c] = ct[c];
            }
        }
    }

    if (X_eval && y_eval && N_eval > 0) {
        int tot_correct = 0, cc[4] = {}, ct[4] = {};
        for (int s_i = 0; s_i < N_eval; s_i++) {
            Tensors::Tensor input({static_cast<size_t>(F), static_cast<size_t>(T)}, FP32);
            for (int t = 0; t < T; t++)
                for (int f = 0; f < F; f++)
                    input.set({static_cast<size_t>(f), static_cast<size_t>(t)},
                              (*X_eval)[(size_t)s_i * T * F + t * F + f]);
            auto out = model.forward(input);
            int pred = 0;
            for (int c = 1; c < NUM_CLASSES; c++)
                if (out.at({(size_t)c}) > out.at({(size_t)pred})) pred = c;
            int yv = (*y_eval)[s_i];
            ct[yv]++;
            if (pred == yv) { tot_correct++; cc[yv]++; }
        }
        L.logf("\nEval: %d/%d = %.1f%%\n", tot_correct, N_eval, 100.0 * tot_correct / N_eval);
        for (int c = 0; c < NUM_CLASSES; c++)
            if (ct[c]) L.logf("  %-8s %3d/%-3d  (%.0f%%)\n", NAMES[c], cc[c], ct[c], 100.0 * cc[c] / ct[c]);

        if (metrics_out) {
            metrics_out->eval_correct = tot_correct;
            metrics_out->eval_total   = N_eval;
            for (int c = 0; c < NUM_CLASSES; c++) {
                metrics_out->eval_cc[c] = cc[c];
                metrics_out->eval_ct[c] = ct[c];
            }
        }
    }
}

static void worker_train(AppState& s) {
    Logger& L = s.log;
    L.log("=== Train ===\n");
    auto tr = DatasetOpener::load_bin(s.train_train_bin);
    L.logf("Loaded %s  N=%d T=%d F=%d\n", s.train_train_bin, tr.N, tr.T, tr.F);
    DatasetOpener::Dataset te{};
    bool have_test = fs::is_regular_file(s.train_test_bin);
    if (have_test) {
        te = DatasetOpener::load_bin(s.train_test_bin);
        L.logf("Loaded %s  N=%d\n", s.train_test_bin, te.N);
    } else {
        L.logf("(%s not found, skipping eval)\n", s.train_test_bin);
    }

    ArchSpec arch{};
    arch.conv_channels = (uint32_t)s.arch_conv_channels;
    arch.conv_kernel   = (uint32_t)s.arch_conv_kernel;
    arch.conv_stride   = (uint32_t)s.arch_conv_stride;
    arch.hidden_layers = (uint32_t)s.arch_hidden_layers;
    arch.hidden_units  = (uint32_t)s.arch_hidden_units;
    L.logf("Arch: conv(out=%u, k=%u, s=%u), hidden=%u x %u units\n",
           arch.conv_channels, arch.conv_kernel, arch.conv_stride,
           arch.hidden_layers, arch.hidden_units);

    auto model = build_model(static_cast<uint32_t>(tr.F), 4u, arch);
    L.logf("Model size: %zu bytes\n\n", model.get_bytes());
    TrainMetrics m{};
    train_loop(s, model, tr.X, tr.y, tr.N, tr.T, tr.F,
               s.train_epochs, s.train_lr,
               have_test ? &te.X : nullptr, have_test ? &te.y : nullptr, have_test ? te.N : 0,
               &m);

    L.log("\nExporting weights...\n");
    export_model_json(model, fs::path(s.train_model_out), &arch);
    L.logf("Wrote weights to %s\n", s.train_model_out);

    // Archive this run so it can be compared against past attempts later.
    archive_run(s, arch, m);

    L.log("=== done ===\n");
    s.runs_refresh = true;  // Runs tab will re-scan the folder on next render
}

// --- Predict worker (CSV -> per-window probabilities) ---
static void worker_predict(AppState& s) {
    Logger& L = s.log;
    L.log("=== Predict ===\n");
    fs::path csv(s.predict_csv);
    if (!fs::is_regular_file(csv)) { L.logf("CSV not found: %s\n", csv.string().c_str()); return; }

    auto rows = read_imu_csv(csv);
    L.logf("CSV: %s  (%zu rows)\n", csv.string().c_str(), rows.size());
    if (rows.size() < 250) { L.log("Need at least 250 rows for one window.\n"); return; }

    auto norm = DatasetOpener::load_bin(s.predict_train_bin);
    L.logf("Normalization from %s  (mean[0]=%.4g)\n", s.predict_train_bin, norm.mean[0]);

    ArchSpec arch = read_arch_from_json(fs::path(s.predict_model));
    auto model = build_model(4u, 4u, arch);
    load_weights_from_json(model, fs::path(s.predict_model));
    L.logf("Loaded weights from %s\n\n", s.predict_model);

    L.log("Window  Rows         idle    walking fidget  drop      -> verdict\n");
    L.log("---------------------------------------------------------------------\n");
    const char* NAMES[4] = {"idle","walking","fidget","drop"};
    int counts[4] = {};
    int n_win = 0;
    for (size_t start = 0; start + 250 <= rows.size(); start += 125) {
        auto input = make_window_tensor(rows, start, norm.mean, norm.std);
        auto probs = model.forward(input);
        int pred = 0;
        for (uint32_t c = 1; c < 4; c++)
            if (probs.at({c}) > probs.at({(size_t)pred})) pred = (int)c;
        L.logf("%-6d  %5zu-%-5zu  %-7.3f %-7.3f %-7.3f %-7.3f   -> %s%s\n",
               n_win, start, start + 250 - 1,
               probs.at({0}), probs.at({1}), probs.at({2}), probs.at({3}),
               NAMES[pred], pred == 3 ? "  <-- DROP" : "");
        counts[pred]++;
        n_win++;
    }
    L.logf("\nSummary across %d windows:\n", n_win);
    for (int c = 0; c < 4; c++)
        L.logf("  %-8s %3d (%.0f%%)\n", NAMES[c], counts[c], n_win ? 100.0 * counts[c] / n_win : 0.0);
    L.log("=== done ===\n");
}

// --- Tune worker (load model.json -> mix user_data with TRAIN.bin -> fine-tune -> write back) ---
static void worker_tune(AppState& s) {
    Logger& L = s.log;
    L.log("=== Tune ===\n");
    auto tr = DatasetOpener::load_bin(s.tune_train_bin);
    L.logf("Loaded original train: N=%d\n", tr.N);

    // Walk user_data/<class>/*.csv with the same window logic.
    std::vector<std::pair<fs::path,int>> user_files;
    int per_class[4] = {};
    for (int lbl = 0; lbl < 4; lbl++) {
        fs::path sub = fs::path(s.tune_user_dir) / dataset_build::CLASS_NAMES[lbl];
        if (!fs::is_directory(sub)) continue;
        for (auto& e : fs::directory_iterator(sub))
            if (e.is_regular_file() && e.path().extension() == ".csv") {
                user_files.emplace_back(e.path(), lbl);
                per_class[lbl]++;
            }
    }
    L.logf("User CSVs: idle=%d walking=%d fidget=%d drop=%d\n",
           per_class[0], per_class[1], per_class[2], per_class[3]);
    if (user_files.empty()) { L.log("No user CSVs found.\n"); return; }

    std::vector<float> X_user;
    std::vector<int32_t> y_user;
    int32_t N_user = 0;
    dataset_build::build_from_files(user_files, X_user, y_user, N_user);
    L.logf("User windows: %d\n", N_user);
    // Apply ORIGINAL normalization from TRAIN.bin -- do not refit.
    dataset_build::apply_normalization(X_user, N_user, tr.T, tr.F, tr.mean, tr.std);

    // Concatenate original + user
    std::vector<float> X_mix(tr.X);
    X_mix.insert(X_mix.end(), X_user.begin(), X_user.end());
    std::vector<int32_t> y_mix(tr.y);
    y_mix.insert(y_mix.end(), y_user.begin(), y_user.end());
    int N_mix = tr.N + N_user;
    L.logf("Mixed: %d (orig %d + user %d)\n", N_mix, tr.N, N_user);

    ArchSpec arch = read_arch_from_json(fs::path(s.tune_model_in));
    L.logf("Arch (from %s): conv(out=%u, k=%u, s=%u), hidden=%u x %u units\n",
           s.tune_model_in, arch.conv_channels, arch.conv_kernel, arch.conv_stride,
           arch.hidden_layers, arch.hidden_units);
    auto model = build_model(static_cast<uint32_t>(tr.F), 4u, arch);
    load_weights_from_json(model, fs::path(s.tune_model_in));
    L.logf("Loaded current weights from %s\n\n", s.tune_model_in);

    DatasetOpener::Dataset te{};
    bool have_test = fs::is_regular_file(s.tune_test_bin);
    if (have_test) te = DatasetOpener::load_bin(s.tune_test_bin);
    train_loop(s, model, X_mix, y_mix, N_mix, tr.T, tr.F,
               s.tune_epochs, s.tune_lr,
               have_test ? &te.X : nullptr, have_test ? &te.y : nullptr, have_test ? te.N : 0);

    fs::path out = s.tune_keep_old ? fs::path("model_tuned.json") : fs::path(s.tune_model_in);
    export_model_json(model, out, &arch);
    L.logf("Wrote tuned weights to %s\n", out.string().c_str());
    if (s.tune_keep_old) L.logf("(original %s preserved)\n", s.tune_model_in);
    L.log("=== done ===\n");
}

// ============================================================================
// Run archive  (writes runs/<timestamp>/{model.json, metrics.json})
// ============================================================================

static std::string timestamp_folder() {
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm; localtime_s(&tm, &tt);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d_%H-%M-%S", &tm);
    return buf;
}
static std::string timestamp_display() {
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm; localtime_s(&tm, &tt);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return buf;
}

static long long count_params(const ArchSpec& a, uint32_t in_channels = 4, uint32_t num_classes = 4) {
    long long conv_p = (long long)in_channels * a.conv_channels * a.conv_kernel + a.conv_channels;
    long long prev = a.conv_channels;
    long long dense_p = 0;
    for (uint32_t i = 0; i < a.hidden_layers; i++) {
        dense_p += prev * a.hidden_units + a.hidden_units;
        prev = a.hidden_units;
    }
    dense_p += prev * num_classes + num_classes;
    return conv_p + dense_p;
}

static void archive_run(AppState& s, const ArchSpec& arch, const TrainMetrics& m) {
    Logger& L = s.log;
    fs::path archive_dir = fs::path(s.runs_dir);
    std::error_code ec;
    fs::create_directories(archive_dir, ec);

    std::string ts_folder = timestamp_folder();
    fs::path run_dir = archive_dir / ts_folder;
    if (!fs::create_directories(run_dir, ec)) {
        L.logf("Could not create run folder %s\n", run_dir.string().c_str());
        return;
    }

    // Snapshot the model weights into the run folder.
    fs::path src_model = fs::path(s.train_model_out);
    fs::path dst_model = run_dir / "model.json";
    if (fs::is_regular_file(src_model)) {
        fs::copy_file(src_model, dst_model, fs::copy_options::overwrite_existing, ec);
    }

    const long long total_p = count_params(arch);

    // Write metrics.json by hand (small, well-known shape).
    fs::path metrics_path = run_dir / "metrics.json";
    std::ofstream f(metrics_path);
    f << "{\n";
    f << "  \"timestamp\": \"" << timestamp_display() << "\",\n";
    f << "  \"note\": \"" << s.train_note << "\",\n";
    f << "  \"arch\": {\n";
    f << "    \"conv_channels\": " << arch.conv_channels << ",\n";
    f << "    \"conv_kernel\": "   << arch.conv_kernel   << ",\n";
    f << "    \"conv_stride\": "   << arch.conv_stride   << ",\n";
    f << "    \"hidden_layers\": " << arch.hidden_layers << ",\n";
    f << "    \"hidden_units\": "  << arch.hidden_units  << "\n";
    f << "  },\n";
    f << "  \"params\": " << total_p << ",\n";
    f << "  \"training\": {\n";
    f << "    \"epochs\": "        << s.train_epochs << ",\n";
    f << "    \"learning_rate\": " << s.train_lr     << ",\n";
    f << "    \"final_loss\": "    << m.final_loss   << ",\n";
    f << "    \"final_train_acc\": " << (m.final_total > 0 ? (double)m.final_correct / m.final_total : 0.0) << "\n";
    f << "  },\n";
    f << "  \"evaluation\": {\n";
    f << "    \"test_correct\": "  << m.eval_correct << ",\n";
    f << "    \"test_total\": "    << m.eval_total   << ",\n";
    f << "    \"test_accuracy\": " << (m.eval_total > 0 ? (double)m.eval_correct / m.eval_total : 0.0) << ",\n";
    f << "    \"per_class\": {\n";
    const char* NAMES[4] = {"idle","walking","fidget","drop"};
    for (int c = 0; c < 4; c++) {
        double acc = m.eval_ct[c] > 0 ? (double)m.eval_cc[c] / m.eval_ct[c] : 0.0;
        f << "      \"" << NAMES[c] << "\": " << acc << (c < 3 ? "," : "") << "\n";
    }
    f << "    }\n";
    f << "  }\n";
    f << "}\n";
    f.close();

    L.logf("Archived run to %s\n", run_dir.string().c_str());
}

// ---- JSON parsing helpers for metrics.json ----

static double json_double_field(const std::string& json,
                                 const std::string& obj_key,
                                 const std::string& field,
                                 double dflt) {
    size_t search_from = 0;
    if (!obj_key.empty() && obj_key != "<root>") {
        const std::string obj_marker = "\"" + obj_key + "\"";
        size_t o = json.find(obj_marker);
        if (o == std::string::npos) return dflt;
        search_from = o;
    }
    const std::string field_marker = "\"" + field + "\"";
    size_t k = json.find(field_marker, search_from);
    if (k == std::string::npos) return dflt;
    size_t colon = json.find(':', k);
    if (colon == std::string::npos) return dflt;
    size_t p = colon + 1;
    while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) p++;
    if (p >= json.size()) return dflt;
    char* end = nullptr;
    double v = std::strtod(json.data() + p, &end);
    if (end == json.data() + p) return dflt;
    return v;
}

static std::string json_string_field(const std::string& json,
                                      const std::string& obj_key,
                                      const std::string& field,
                                      const std::string& dflt) {
    const std::string obj_marker = "\"" + obj_key + "\"";
    size_t search_from = (obj_key == "<root>") ? 0 : json.find(obj_marker);
    if (search_from == std::string::npos) return dflt;
    const std::string field_marker = "\"" + field + "\"";
    size_t k = json.find(field_marker, search_from);
    if (k == std::string::npos) return dflt;
    size_t colon = json.find(':', k);
    if (colon == std::string::npos) return dflt;
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return dflt;
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return dflt;
    return json.substr(q1 + 1, q2 - q1 - 1);
}

static std::vector<RunInfo> scan_runs(const fs::path& dir) {
    std::vector<RunInfo> out;
    if (!fs::is_directory(dir)) return out;
    const char* NAMES[4] = {"idle","walking","fidget","drop"};
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_directory()) continue;
        fs::path mpath = e.path() / "metrics.json";
        if (!fs::is_regular_file(mpath)) continue;
        std::string j;
        try { j = read_text_file(mpath); } catch (...) { continue; }

        RunInfo r;
        r.folder = e.path();
        r.timestamp = json_string_field(j, "<root>", "timestamp",
                                        e.path().filename().string());
        r.note      = json_string_field(j, "<root>", "note", "");
        r.arch.conv_channels = (uint32_t)json_int_field(j, "arch", "conv_channels", 8);
        r.arch.conv_kernel   = (uint32_t)json_int_field(j, "arch", "conv_kernel",   7);
        r.arch.conv_stride   = (uint32_t)json_int_field(j, "arch", "conv_stride",   2);
        r.arch.hidden_layers = (uint32_t)json_int_field(j, "arch", "hidden_layers", 0);
        r.arch.hidden_units  = (uint32_t)json_int_field(j, "arch", "hidden_units",  16);
        r.params           = (long long)json_int_field(j, "<root>", "params", 0);
        r.epochs           = json_int_field(j, "training", "epochs", 0);
        r.learning_rate    = json_double_field(j, "training", "learning_rate", 0.0);
        r.final_loss       = json_double_field(j, "training", "final_loss",      0.0);
        r.final_train_acc  = json_double_field(j, "training", "final_train_acc", 0.0);
        r.eval_correct     = json_int_field   (j, "evaluation", "test_correct", 0);
        r.eval_total       = json_int_field   (j, "evaluation", "test_total",   0);
        r.test_accuracy    = json_double_field(j, "evaluation", "test_accuracy", 0.0);
        for (int c = 0; c < 4; c++)
            r.per_class_acc[c] = json_double_field(j, "per_class", NAMES[c], 0.0);

        out.push_back(std::move(r));
    }
    // Newest first by folder name (timestamp-ordered).
    std::sort(out.begin(), out.end(),
              [](const RunInfo& a, const RunInfo& b) {
                  return a.folder.filename().string() > b.folder.filename().string();
              });
    return out;
}

// ============================================================================
// Small UI helpers
// ============================================================================
static ImVec4 rgb(int r, int g, int b, float a = 1.0f) {
    return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a);
}

static void status_mark(bool ok, const char* label) {
    ImVec2 p = ImGui::GetCursorScreenPos();
    float h = ImGui::GetTextLineHeight();
    float r = 5.0f;
    ImU32 col  = ok ? IM_COL32( 80, 200, 120, 255) : IM_COL32(225,  95,  95, 255);
    ImU32 ring = ok ? IM_COL32( 40, 120,  70, 255) : IM_COL32(150,  55,  55, 255);
    ImVec2 center(p.x + r + 4, p.y + h * 0.5f);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddCircleFilled(center, r, col);
    dl->AddCircle(center, r, ring, 0, 1.0f);
    ImGui::Dummy(ImVec2(r * 2 + 12, h));
    ImGui::SameLine();
    if (ok) ImGui::TextUnformatted(label);
    else    ImGui::TextDisabled("%s", label);
}

static void section_header(const char* label) {
    if (g_font_h2) ImGui::PushFont(g_font_h2);
    ImGui::TextUnformatted(label);
    if (g_font_h2) ImGui::PopFont();
    ImGui::Spacing();
}

// Clickable card header. Toggles collapse state for the card identified by `id`.
// Returns true when the card body should be rendered (i.e. card is expanded).
//   card_begin("##my");
//   if (card_header(s, "##my", "Title")) { ...body... }
//   card_end();
static bool card_header(AppState& s, const char* id, const char* label) {
    // Cards default to collapsed on first touch; subsequent toggles persist for the session.
    auto it = s.card_collapsed.find(id);
    if (it == s.card_collapsed.end()) s.card_collapsed.emplace(id, true);
    bool& collapsed = s.card_collapsed[id];
    const float scale = g_dpi_scale;

    // Inner content width inside the card (we're already inside the card child).
    float avail_w = ImGui::GetContentRegionAvail().x;
    if (g_font_h2) ImGui::PushFont(g_font_h2);
    float row_h = ImGui::GetTextLineHeight() + 8.0f * scale;
    if (g_font_h2) ImGui::PopFont();

    ImVec2 row_tl = ImGui::GetCursorScreenPos();
    ImVec2 row_br(row_tl.x + avail_w, row_tl.y + row_h);

    ImGui::PushID(id);
    bool clicked = ImGui::InvisibleButton("##cardh", ImVec2(avail_w, row_h));
    bool hovered = ImGui::IsItemHovered();
    ImGui::PopID();
    if (clicked) collapsed = !collapsed;

    ImDrawList* dl = ImGui::GetWindowDrawList();
    if (hovered) {
        dl->AddRectFilled(row_tl, row_br, IM_COL32(255, 255, 255, 14), 6.0f * scale);
    }

    // Chevron triangle (right when collapsed, down when expanded).
    const ImU32 chev_col = IM_COL32(170, 180, 200, 255);
    const float cs = 5.0f * scale;
    ImVec2 cc(row_tl.x + 10.0f * scale, row_tl.y + row_h * 0.5f);
    if (collapsed) {
        dl->AddTriangleFilled(
            ImVec2(cc.x - cs * 0.6f, cc.y - cs),
            ImVec2(cc.x - cs * 0.6f, cc.y + cs),
            ImVec2(cc.x + cs * 0.9f, cc.y),
            chev_col);
    } else {
        dl->AddTriangleFilled(
            ImVec2(cc.x - cs,        cc.y - cs * 0.6f),
            ImVec2(cc.x + cs,        cc.y - cs * 0.6f),
            ImVec2(cc.x,             cc.y + cs * 0.9f),
            chev_col);
    }

    // Title label
    if (g_font_h2) ImGui::PushFont(g_font_h2);
    float text_h = ImGui::GetTextLineHeight();
    ImVec2 text_pos(row_tl.x + 30.0f * scale, row_tl.y + (row_h - text_h) * 0.5f);
    dl->AddText(text_pos, IM_COL32(225, 228, 234, 255), label);
    if (g_font_h2) ImGui::PopFont();

    // Reserve the row height for following content / spacing.
    if (!collapsed) ImGui::Dummy(ImVec2(0, 4.0f * scale));

    return !collapsed;
}

// Card container -- visually groups a set of widgets in a slightly raised panel.
static void card_begin(const char* id) {
    // Wider side padding, modest top/bottom.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(26.0f * g_dpi_scale, 14.0f * g_dpi_scale));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, rgb(24, 28, 36));
    ImGui::PushStyleColor(ImGuiCol_Border,  rgb(38, 44, 56));
    ImGui::BeginChild(id, ImVec2(0, 0),
                      ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_Border,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
}

static void card_end() {
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar();
}

// Renders a 4-element status grid as 2 rows of 2 columns.
// `info_for` returns (present, "<label>  N csv") for class index 0..3.
template <class F>
static void status_grid_2x2(F info_for) {
    float cell_w = ImGui::GetContentRegionAvail().x * 0.5f;
    for (int row = 0; row < 2; row++) {
        float row_x = ImGui::GetCursorPosX();
        char tmp[128];
        bool present;

        present = info_for(row * 2, tmp, sizeof(tmp));
        status_mark(present, tmp);

        ImGui::SameLine();
        ImGui::SetCursorPosX(row_x + cell_w);
        present = info_for(row * 2 + 1, tmp, sizeof(tmp));
        status_mark(present, tmp);
    }
}

// ----- File / folder pickers (native Windows dialogs) ---------------------------------
static HWND g_main_hwnd = nullptr;  // set in WinMain after CreateWindowW

static std::wstring utf8_to_wide(const char* s) {
    if (!s || !*s) return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
    return w;
}

static fs::path initial_dir(const char* current_value) {
    fs::path p(current_value ? current_value : "");
    if (!p.empty()) {
        if (fs::is_directory(p)) return fs::absolute(p);
        if (p.has_parent_path() && fs::is_directory(p.parent_path())) return fs::absolute(p.parent_path());
    }
    return fs::current_path();
}

static bool pick_folder(char* buf, size_t buf_size) {
    IFileOpenDialog* dlg = nullptr;
    if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(&dlg)))) return false;
    DWORD opts = 0;
    dlg->GetOptions(&opts);
    dlg->SetOptions(opts | FOS_PICKFOLDERS | FOS_PATHMUSTEXIST | FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR);

    auto start = initial_dir(buf);
    IShellItem* psi_start = nullptr;
    if (SUCCEEDED(SHCreateItemFromParsingName(start.wstring().c_str(), nullptr, IID_PPV_ARGS(&psi_start)))) {
        dlg->SetFolder(psi_start);
        psi_start->Release();
    }

    bool ok = false;
    if (SUCCEEDED(dlg->Show(g_main_hwnd))) {
        IShellItem* item = nullptr;
        if (SUCCEEDED(dlg->GetResult(&item))) {
            PWSTR wpath = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &wpath))) {
                int n = WideCharToMultiByte(CP_UTF8, 0, wpath, -1, buf, (int)buf_size, nullptr, nullptr);
                ok = n > 0;
                CoTaskMemFree(wpath);
            }
            item->Release();
        }
    }
    dlg->Release();
    return ok;
}

static bool pick_file_common(bool save, char* buf, size_t buf_size, const char* filter) {
    OPENFILENAMEA ofn{};
    char tmp[1024]; std::memset(tmp, 0, sizeof(tmp));
    if (buf && *buf) std::strncpy(tmp, buf, sizeof(tmp) - 1);

    auto init_dir = initial_dir(buf);
    std::string init_dir_str = init_dir.string();

    ofn.lStructSize  = sizeof(ofn);
    ofn.hwndOwner    = g_main_hwnd;
    ofn.lpstrFile    = tmp;
    ofn.nMaxFile     = sizeof(tmp);
    ofn.lpstrFilter  = filter ? filter : "All files\0*.*\0";
    ofn.lpstrInitialDir = init_dir_str.c_str();
    ofn.Flags = OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_EXPLORER;
    if (save) ofn.Flags |= OFN_OVERWRITEPROMPT;
    else      ofn.Flags |= OFN_FILEMUSTEXIST;

    BOOL r = save ? GetSaveFileNameA(&ofn) : GetOpenFileNameA(&ofn);
    if (!r) return false;
    std::strncpy(buf, tmp, buf_size - 1);
    buf[buf_size - 1] = '\0';
    return true;
}
static bool pick_file_open(char* buf, size_t buf_size, const char* filter) { return pick_file_common(false, buf, buf_size, filter); }
static bool pick_file_save(char* buf, size_t buf_size, const char* filter) { return pick_file_common(true,  buf, buf_size, filter); }

enum class PickKind { Folder, FileOpen, FileSave };

// Labeled path field: label on top, [input field][Browse] on the row below.
static void labeled_path(const char* id, const char* label,
                         char* buf, size_t buf_size,
                         PickKind kind, const char* filter = nullptr) {
    ImGui::PushStyleColor(ImGuiCol_Text, rgb(155, 165, 180));
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();

    const float btn_w = 84.0f * g_dpi_scale;
    const float gap   = 6.0f  * g_dpi_scale;
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - btn_w - gap);
    ImGui::InputText(id, buf, buf_size);
    ImGui::PopItemWidth();
    ImGui::SameLine(0, gap);

    char btn_id[64];
    std::snprintf(btn_id, sizeof(btn_id), "Browse%s", id);  // id already has "##"
    if (ImGui::Button(btn_id, ImVec2(btn_w, 0))) {
        switch (kind) {
            case PickKind::Folder:   pick_folder(buf, buf_size); break;
            case PickKind::FileOpen: pick_file_open(buf, buf_size, filter); break;
            case PickKind::FileSave: pick_file_save(buf, buf_size, filter); break;
        }
    }
    ImGui::Dummy(ImVec2(0, 2.0f * g_dpi_scale));
}

// Labeled input field with the label above (modern form layout).
static void labeled_text(const char* id, const char* label, char* buf, size_t buf_size) {
    ImGui::PushStyleColor(ImGuiCol_Text, rgb(155, 165, 180));
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::PushItemWidth(-FLT_MIN);
    ImGui::InputText(id, buf, buf_size);
    ImGui::PopItemWidth();
    ImGui::Dummy(ImVec2(0, 2.0f * g_dpi_scale));
}

// Numeric input with +/- steppers, full-width.
static void labeled_int(const char* id, const char* label, int* v, int step = 1, int step_fast = 5) {
    ImGui::PushStyleColor(ImGuiCol_Text, rgb(155, 165, 180));
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::PushItemWidth(-FLT_MIN);
    ImGui::InputInt(id, v, step, step_fast);
    ImGui::PopItemWidth();
    ImGui::Dummy(ImVec2(0, 2.0f * g_dpi_scale));
}

static void labeled_float(const char* id, const char* label, float* v,
                          float step = 0.0001f, float step_fast = 0.001f, const char* fmt = "%.4f") {
    ImGui::PushStyleColor(ImGuiCol_Text, rgb(155, 165, 180));
    ImGui::TextUnformatted(label);
    ImGui::PopStyleColor();
    ImGui::PushItemWidth(-FLT_MIN);
    ImGui::InputFloat(id, v, step, step_fast, fmt);
    ImGui::PopItemWidth();
    ImGui::Dummy(ImVec2(0, 2.0f * g_dpi_scale));
}

static int count_csvs(const fs::path& dir) {
    if (!fs::is_directory(dir)) return -1;
    int n = 0;
    for (auto& e : fs::directory_iterator(dir))
        if (e.is_regular_file() && e.path().extension() == ".csv") n++;
    return n;
}

// Light syntax-coloring for the log: section headers, errors, completions.
// Decides the color for one log line based on its leading content.
static bool log_line_color(const char* p, const char* end, ImVec4& out) {
    auto starts_with = [&](const char* tag) {
        size_t n = std::strlen(tag);
        return (size_t)(end - p) >= n && std::strncmp(p, tag, n) == 0;
    };
    if ((end - p) >= 3 && p[0] == '=' && p[1] == '=' && p[2] == '=') { out = rgb(110, 175, 250); return true; }
    if (starts_with("ERROR"))                                          { out = rgb(235, 100, 100); return true; }
    if (starts_with("Wrote") || starts_with("Saved"))                  { out = rgb(110, 200, 130); return true; }
    if (starts_with("Epoch"))                                          { out = rgb(200, 215, 240); return true; }
    if (starts_with("//") || starts_with("static const"))              { out = rgb(150, 200, 130); return true; }
    if (starts_with("Test accuracy") || starts_with("Eval"))           { out = rgb(160, 220, 160); return true; }
    if (starts_with("  ") && std::strstr(p, "%") &&
        (end - p) > 6 && (p[2] == 'i' || p[2] == 'w' || p[2] == 'f' || p[2] == 'd')) { out = rgb(200, 200, 220); return true; }
    return false;
}

static void render_log_panel(AppState& s, float height) {
    s.log.snapshot(s.log_snapshot);

    ImGui::PushStyleColor(ImGuiCol_ChildBg, rgb(14, 17, 22));
    ImGui::PushStyleColor(ImGuiCol_Border,  rgb(34, 40, 52));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                        ImVec2(18.0f * g_dpi_scale, 14.0f * g_dpi_scale));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f * g_dpi_scale);
    // No horizontal scrollbar -- long lines wrap to the panel width instead.
    ImGui::BeginChild("log_panel", ImVec2(0, height), ImGuiChildFlags_Border,
                      ImGuiWindowFlags_None);

    if (g_font_mono) ImGui::PushFont(g_font_mono);

    // Wrap text at the right edge of the panel so lines never run off-screen.
    ImGui::PushTextWrapPos(0.0f);

    if (s.log_snapshot.empty()) {
        // Center a faint placeholder.
        ImGui::PushStyleColor(ImGuiCol_Text, rgb(85, 95, 115));
        const char* msg = "(no output yet — press a run button)";
        ImVec2 sz = ImGui::CalcTextSize(msg);
        ImVec2 avail = ImGui::GetContentRegionAvail();
        ImGui::SetCursorPos(ImVec2((avail.x - sz.x) * 0.5f, (avail.y - sz.y) * 0.5f));
        ImGui::TextUnformatted(msg);
        ImGui::PopStyleColor();
    } else {
        // Tighter line spacing for log readability.
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 2.0f * g_dpi_scale));
        const char* p = s.log_snapshot.c_str();
        const char* end = p + s.log_snapshot.size();
        while (p < end) {
            const char* nl = (const char*)std::memchr(p, '\n', end - p);
            const char* line_end = nl ? nl : end;
            ImVec4 col;
            bool have_col = log_line_color(p, line_end, col);
            if (have_col) ImGui::PushStyleColor(ImGuiCol_Text, col);
            if (line_end == p) ImGui::Dummy(ImVec2(0, ImGui::GetTextLineHeight()));
            else               ImGui::TextUnformatted(p, line_end);
            if (have_col) ImGui::PopStyleColor();
            p = nl ? nl + 1 : end;
        }
        ImGui::PopStyleVar();

        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 60.0f)
            ImGui::SetScrollHereY(1.0f);
    }

    ImGui::PopTextWrapPos();
    if (g_font_mono) ImGui::PopFont();
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
}

static void run_button(AppState& s, const char* label, std::function<void()> action) {
    bool busy = s.busy.load();
    // Width fixed in DPI-aware px, height = 0 so ImGui auto-fits text+padding (perfectly centered text).
    const ImVec2 primary  { 180.0f * g_dpi_scale, 0.0f };
    const ImVec2 secondary{ 110.0f * g_dpi_scale, 0.0f };
    const float  gap      =  10.0f * g_dpi_scale;
    const float  total_w  = primary.x + gap + secondary.x;

    // Horizontally center the button row in the available column width.
    float avail = ImGui::GetContentRegionAvail().x;
    if (avail > total_w) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - total_w) * 0.5f);
    }

    if (busy) ImGui::BeginDisabled();
    if (ImGui::Button(label, primary)) {
        s.log.clear();
        launch_worker(s, std::move(action));
    }
    if (busy) ImGui::EndDisabled();
    ImGui::SameLine(0, gap);
    if (ImGui::Button("Clear log", secondary)) s.log.clear();
}

// ============================================================================
// Burger navigation -- single button, opens a popup with the four section names.
// ============================================================================
static void draw_burger_nav(AppState& s) {
    const float scale     = g_dpi_scale;
    const float btn_size  = 38.0f * scale;
    const float row_h     = btn_size;

    ImVec2 btn_pos = ImGui::GetCursorScreenPos();

    // The burger itself is an invisible button so we own the icon rendering.
    ImGui::PushID("##nav");
    bool clicked = ImGui::InvisibleButton("burger", ImVec2(btn_size, btn_size));
    bool hovered = ImGui::IsItemHovered();
    ImGui::PopID();

    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImU32 fill_normal = IM_COL32(32, 38, 50, 255);
    ImU32 fill_hover  = IM_COL32(46, 56, 76, 255);
    ImU32 fill_active = IM_COL32(60, 110, 180, 255);
    bool open = ImGui::IsPopupOpen("##nav_popup");
    ImU32 fill = open ? fill_active : (hovered ? fill_hover : fill_normal);
    dl->AddRectFilled(btn_pos, ImVec2(btn_pos.x + btn_size, btn_pos.y + btn_size),
                      fill, 8.0f * scale);

    // Three horizontal lines (burger icon) centered in the button.
    ImVec2 c(btn_pos.x + btn_size * 0.5f, btn_pos.y + btn_size * 0.5f);
    float  half_w = 8.0f * scale;
    float  spacing = 5.0f * scale;
    float  thick = 1.6f * scale;
    ImU32  ink = IM_COL32(225, 228, 234, 255);
    dl->AddLine(ImVec2(c.x - half_w, c.y - spacing), ImVec2(c.x + half_w, c.y - spacing), ink, thick);
    dl->AddLine(ImVec2(c.x - half_w, c.y),           ImVec2(c.x + half_w, c.y),           ink, thick);
    dl->AddLine(ImVec2(c.x - half_w, c.y + spacing), ImVec2(c.x + half_w, c.y + spacing), ink, thick);

    if (clicked) ImGui::OpenPopup("##nav_popup");

    // Current section name next to the burger.
    ImGui::SameLine(0, 14.0f * scale);
    if (g_font_h2) ImGui::PushFont(g_font_h2);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + (row_h - ImGui::GetTextLineHeight()) * 0.5f);
    ImGui::TextUnformatted(tab_label(s.active_tab));
    if (g_font_h2) ImGui::PopFont();

    // Popup anchored just below the burger, with custom item rendering for polish.
    ImGui::SetNextWindowPos(ImVec2(btn_pos.x, btn_pos.y + btn_size + 6.0f * scale));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f * scale, 8.0f * scale));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,   ImVec2(0, 4.0f * scale));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 10.0f * scale);
    // Distinctly lighter than the window background so the popup reads as a panel
    // floating above the app, plus a brighter accent border.
    ImGui::PushStyleColor(ImGuiCol_PopupBg, rgb(42, 50, 66));
    ImGui::PushStyleColor(ImGuiCol_Border,  rgb(96, 130, 180));
    if (ImGui::BeginPopup("##nav_popup")) {
        const float item_w  = 240.0f * scale;
        const float item_h  = 40.0f  * scale;
        const float radius  = 6.0f   * scale;
        const float pad_l   = 18.0f  * scale;
        const float bar_w   = 3.0f   * scale;
        ImDrawList* pdl = ImGui::GetWindowDrawList();

        for (int i = 0; i < NUM_TABS; i++) {
            Tab t = (Tab)i;
            bool active = (s.active_tab == t);
            ImVec2 ip = ImGui::GetCursorScreenPos();
            ImGui::PushID(i);
            bool clicked = ImGui::InvisibleButton("##item", ImVec2(item_w, item_h));
            bool hovered = ImGui::IsItemHovered();
            ImGui::PopID();

            // Background fill (tuned to read clearly against the lighter popup bg).
            ImU32 bg = 0;
            if      (active && hovered) bg = IM_COL32(64, 120, 200, 255);
            else if (active)            bg = IM_COL32(50,  95, 160, 255);
            else if (hovered)           bg = IM_COL32(60,  72,  92, 255);
            if (bg) pdl->AddRectFilled(ip, ImVec2(ip.x + item_w, ip.y + item_h), bg, radius);

            // Left accent bar for the active item
            if (active) {
                pdl->AddRectFilled(ImVec2(ip.x + 6 * scale, ip.y + 8 * scale),
                                   ImVec2(ip.x + 6 * scale + bar_w, ip.y + item_h - 8 * scale),
                                   IM_COL32(110, 175, 250, 255), bar_w);
            }

            // Label (vertically centered)
            const char* label = tab_label(t);
            float text_h = ImGui::GetTextLineHeight();
            ImU32 text_col = active ? IM_COL32(245, 248, 252, 255) : IM_COL32(220, 225, 232, 255);
            pdl->AddText(ImVec2(ip.x + pad_l, ip.y + (item_h - text_h) * 0.5f), text_col, label);

            if (clicked) {
                s.active_tab = t;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(3);
}

// ============================================================================
// Tabs -- responsive 2-column layout that collapses to 1 column when narrow.
// ============================================================================
static void two_col_begin(float& left_w_out, float& gap_out) {
    gap_out    = 14.0f * g_dpi_scale;
    float avail = ImGui::GetContentRegionAvail().x;
    left_w_out  = std::clamp(avail * 0.40f, 380.0f * g_dpi_scale, 520.0f * g_dpi_scale);
}

// Layout helper: arranges every tab's three sections.
//   wide  : LEFT  -> [inputs card]
//                    [status card]
//                    [run button row]
//           RIGHT -> [output log (full column height)]
//   narrow: [inputs card]
//           [status card]
//           [run button]
//           [output log]
template <class FInputs, class FRun, class FStatus>
static void tab_layout(AppState& s, FInputs inputs_card, FRun run_btn, FStatus status_card) {
    float avail = ImGui::GetContentRegionAvail().x;
    bool narrow = avail < NARROW_PX_UNSCALED * g_dpi_scale;

    if (narrow) {
        inputs_card();
        ImGui::Dummy(ImVec2(0, 6.0f * g_dpi_scale));
        status_card();
        ImGui::Dummy(ImVec2(0, 10.0f * g_dpi_scale));
        run_btn();
        ImGui::Dummy(ImVec2(0, 10.0f * g_dpi_scale));
        section_header("Output");
        render_log_panel(s, -1.0f);
        return;
    }

    // ----- Wide layout -----
    float left_w, gap;
    two_col_begin(left_w, gap);
    float col_h = ImGui::GetContentRegionAvail().y;

    // LEFT column: stacked cards + run button at the bottom (no scrollbar).
    ImGui::BeginChild("##col_left", ImVec2(left_w, col_h), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        inputs_card();
        ImGui::Dummy(ImVec2(0, 6.0f * g_dpi_scale));
        status_card();
        ImGui::Dummy(ImVec2(0, 10.0f * g_dpi_scale));
        run_btn();
    }
    ImGui::EndChild();

    ImGui::SameLine(0, gap);

    // RIGHT column: output log spans the full column height.
    ImGui::BeginChild("##col_right", ImVec2(0, col_h), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    {
        section_header("Output");
        render_log_panel(s, -1.0f);
    }
    ImGui::EndChild();
}

static void render_build_tab(AppState& s) {
    tab_layout(s,
        [&] {
            card_begin("##build_inputs");
            if (card_header(s, "##build_inputs", "Inputs")) {
                labeled_path("##b_src",   "Source folder",  s.build_src,   sizeof(s.build_src),   PickKind::Folder);
                labeled_path("##b_train", "Output train",   s.build_train, sizeof(s.build_train), PickKind::FileSave, "Binary dataset\0*.bin\0All files\0*.*\0");
                labeled_path("##b_test",  "Output test",    s.build_test,  sizeof(s.build_test),  PickKind::FileSave, "Binary dataset\0*.bin\0All files\0*.*\0");
                labeled_int ("##b_pct",   "Test split (%)", &s.build_test_pct, 5, 25);
                labeled_int ("##b_seed",  "Random seed",    &s.build_seed,  1, 10);
            }
            card_end();
        },
        [&] { run_button(s, "Build dataset", [&s] { worker_build_dataset(s); }); },
        [&] {
            card_begin("##build_status");
            if (card_header(s, "##build_status", "Source files")) {
                status_grid_2x2([&](int c, char* buf, size_t buf_n) {
                    int n = count_csvs(fs::path(s.build_src) / dataset_build::CLASS_NAMES[c]);
                    if (n < 0) std::snprintf(buf, buf_n, "%-7s  --",     dataset_build::CLASS_NAMES[c]);
                    else       std::snprintf(buf, buf_n, "%-7s  %d csv", dataset_build::CLASS_NAMES[c], n);
                    return n > 0;
                });
            }
            card_end();
        });
}

static void render_train_tab(AppState& s) {
    tab_layout(s,
        [&] {
            card_begin("##train_inputs");
            if (card_header(s, "##train_inputs", "Inputs")) {
                labeled_path ("##t_train", "TRAIN.bin",     s.train_train_bin, sizeof(s.train_train_bin), PickKind::FileOpen, "Binary dataset\0*.bin\0All files\0*.*\0");
                labeled_path ("##t_test",  "TEST.bin",      s.train_test_bin,  sizeof(s.train_test_bin),  PickKind::FileOpen, "Binary dataset\0*.bin\0All files\0*.*\0");
                labeled_path ("##t_out",   "Output JSON",   s.train_model_out, sizeof(s.train_model_out), PickKind::FileSave, "JSON model\0*.json\0All files\0*.*\0");
                labeled_int  ("##t_ep",    "Epochs",        &s.train_epochs, 1, 5);
                labeled_float("##t_lr",    "Learning rate", &s.train_lr, 0.0005f, 0.005f, "%.4f");
                labeled_text ("##t_note",  "Run note (optional)", s.train_note, sizeof(s.train_note));
            }
            card_end();
        },
        [&] { run_button(s, "Train model", [&s] { worker_train(s); }); },
        [&] {
            // ---- Architecture card ----
            card_begin("##train_arch");
            if (card_header(s, "##train_arch", "Architecture")) {
                labeled_int("##a_conv_ch", "Conv1D channels (output)", &s.arch_conv_channels, 1, 4);
                labeled_int("##a_k",       "Conv1D kernel size",       &s.arch_conv_kernel,   2, 2);
                labeled_int("##a_s",       "Conv1D stride",            &s.arch_conv_stride,   1, 1);
                labeled_int("##a_hl",      "Hidden Dense layers",      &s.arch_hidden_layers, 1, 1);
                if (s.arch_hidden_layers > 0)
                    labeled_int("##a_hu",  "Units per hidden layer",   &s.arch_hidden_units,  4, 8);
                // Clamp to sane ranges so users can't enter garbage that crashes the model builder.
                s.arch_conv_channels = std::clamp(s.arch_conv_channels, 2,  64);
                s.arch_conv_kernel   = std::clamp(s.arch_conv_kernel,   2,  21);
                s.arch_conv_stride   = std::clamp(s.arch_conv_stride,   1,  16);
                s.arch_hidden_layers = std::clamp(s.arch_hidden_layers, 0,   3);
                s.arch_hidden_units  = std::clamp(s.arch_hidden_units,  2, 256);
                {
                    // Footprint readout: # params, useful for keeping the Flipper budget in mind.
                    uint32_t T_in = 249;
                    uint32_t L_out = (s.arch_conv_kernel <= T_in) ? (T_in - s.arch_conv_kernel) / std::max(1, s.arch_conv_stride) + 1 : 0u;
                    long long conv_p = 4LL * s.arch_conv_channels * s.arch_conv_kernel + s.arch_conv_channels;
                    long long prev = s.arch_conv_channels;
                    long long dense_p = 0;
                    for (int i = 0; i < s.arch_hidden_layers; i++) {
                        dense_p += prev * s.arch_hidden_units + s.arch_hidden_units;
                        prev = s.arch_hidden_units;
                    }
                    dense_p += prev * 4 + 4;
                    long long total = conv_p + dense_p;
                    ImGui::PushStyleColor(ImGuiCol_Text, rgb(140, 150, 170));
                    ImGui::Text("Params: %lld  ~ %lld bytes (FP32)", total, total * 4);
                    ImGui::Text("Conv1D out length: %u", L_out);
                    ImGui::PopStyleColor();
                }
            }
            card_end();

            // ---- Required files ----
            card_begin("##train_status");
            if (card_header(s, "##train_status", "Required files")) {
                status_mark(fs::is_regular_file(s.train_train_bin), s.train_train_bin);
                status_mark(fs::is_regular_file(s.train_test_bin),  s.train_test_bin);
            }
            card_end();
        });
}

static void render_predict_tab(AppState& s) {
    tab_layout(s,
        [&] {
            card_begin("##pred_inputs");
            if (card_header(s, "##pred_inputs", "Inputs")) {
                labeled_path("##p_csv",   "CSV path",   s.predict_csv,       sizeof(s.predict_csv),       PickKind::FileOpen, "CSV files\0*.csv\0All files\0*.*\0");
                labeled_path("##p_model", "model.json", s.predict_model,     sizeof(s.predict_model),     PickKind::FileOpen, "JSON model\0*.json\0All files\0*.*\0");
                labeled_path("##p_train", "TRAIN.bin",  s.predict_train_bin, sizeof(s.predict_train_bin), PickKind::FileOpen, "Binary dataset\0*.bin\0All files\0*.*\0");
            }
            card_end();
        },
        [&] { run_button(s, "Run predict", [&s] { worker_predict(s); }); },
        [&] {
            card_begin("##pred_status");
            if (card_header(s, "##pred_status", "Required files")) {
                status_mark(fs::is_regular_file(s.predict_csv),       s.predict_csv[0] ? s.predict_csv : "(no CSV selected)");
                status_mark(fs::is_regular_file(s.predict_model),     s.predict_model);
                status_mark(fs::is_regular_file(s.predict_train_bin), s.predict_train_bin);
            }
            card_end();
        });
}

static void render_tune_tab(AppState& s) {
    bool any_user = false;
    for (int c = 0; c < 4 && !any_user; c++)
        if (count_csvs(fs::path(s.tune_user_dir) / dataset_build::CLASS_NAMES[c]) > 0) any_user = true;
    const bool gate = any_user
                   && fs::is_regular_file(s.tune_model_in)
                   && fs::is_regular_file(s.tune_train_bin);

    tab_layout(s,
        [&] {
            card_begin("##tune_inputs");
            if (card_header(s, "##tune_inputs", "Inputs")) {

            // Two sub-columns: paths on the left (need width), hyperparams + checkbox on the right.
            float avail_w = ImGui::GetContentRegionAvail().x;
            float sub_gap = 22.0f * g_dpi_scale;
            float right_w = 200.0f * g_dpi_scale;
            float left_w  = avail_w - right_w - sub_gap;
            if (left_w < 260.0f * g_dpi_scale) {       // narrow card -> fall back to single column
                labeled_path ("##u_dir",   "user_data folder", s.tune_user_dir,  sizeof(s.tune_user_dir),  PickKind::Folder);
                labeled_path ("##u_model", "model.json",       s.tune_model_in,  sizeof(s.tune_model_in),  PickKind::FileOpen, "JSON model\0*.json\0All files\0*.*\0");
                labeled_path ("##u_train", "TRAIN.bin",        s.tune_train_bin, sizeof(s.tune_train_bin), PickKind::FileOpen, "Binary dataset\0*.bin\0All files\0*.*\0");
                labeled_path ("##u_test",  "TEST.bin",         s.tune_test_bin,  sizeof(s.tune_test_bin),  PickKind::FileOpen, "Binary dataset\0*.bin\0All files\0*.*\0");
                labeled_int  ("##u_ep",    "Epochs",           &s.tune_epochs, 1, 5);
                labeled_float("##u_lr",    "Learning rate",    &s.tune_lr, 0.0005f, 0.005f, "%.4f");
                ImGui::Checkbox("Keep old (write to model_tuned.json)", &s.tune_keep_old);
            } else {
                ImGui::BeginChild("##tune_paths", ImVec2(left_w, 0),
                                  ImGuiChildFlags_AutoResizeY,
                                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                labeled_path ("##u_dir",   "user_data folder", s.tune_user_dir,  sizeof(s.tune_user_dir),  PickKind::Folder);
                labeled_path ("##u_model", "model.json",       s.tune_model_in,  sizeof(s.tune_model_in),  PickKind::FileOpen, "JSON model\0*.json\0All files\0*.*\0");
                labeled_path ("##u_train", "TRAIN.bin",        s.tune_train_bin, sizeof(s.tune_train_bin), PickKind::FileOpen, "Binary dataset\0*.bin\0All files\0*.*\0");
                labeled_path ("##u_test",  "TEST.bin",         s.tune_test_bin,  sizeof(s.tune_test_bin),  PickKind::FileOpen, "Binary dataset\0*.bin\0All files\0*.*\0");
                ImGui::EndChild();

                ImGui::SameLine(0, sub_gap);

                ImGui::BeginChild("##tune_hp", ImVec2(right_w, 0),
                                  ImGuiChildFlags_AutoResizeY,
                                  ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                labeled_int  ("##u_ep",    "Epochs",        &s.tune_epochs, 1, 5);
                labeled_float("##u_lr",    "Learning rate", &s.tune_lr, 0.0005f, 0.005f, "%.4f");
                ImGui::Dummy(ImVec2(0, 6.0f * g_dpi_scale));
                ImGui::Checkbox("Keep old", &s.tune_keep_old);
                ImGui::EndChild();
            }
            }  // end of if (card_header ... Inputs)
            card_end();
        },
        [&] {
            if (!gate) ImGui::BeginDisabled();
            run_button(s, "Tune model", [&s] { worker_tune(s); });
            if (!gate) ImGui::EndDisabled();
        },
        [&] {
            card_begin("##tune_user");
            if (card_header(s, "##tune_user", "User data")) {
            status_grid_2x2([&](int c, char* buf, size_t buf_n) {
                int n = count_csvs(fs::path(s.tune_user_dir) / dataset_build::CLASS_NAMES[c]);
                if (n < 0) std::snprintf(buf, buf_n, "%-7s  --",     dataset_build::CLASS_NAMES[c]);
                else       std::snprintf(buf, buf_n, "%-7s  %d csv", dataset_build::CLASS_NAMES[c], n);
                return n > 0;
            });
            }
            card_end();
        });
}

// ============================================================================
// Runs tab -- archived training runs, sortable, with Load / Delete actions.
// ============================================================================
static void render_runs_tab(AppState& s) {
    static std::vector<RunInfo> runs_cache;
    if (s.runs_refresh) {
        runs_cache = scan_runs(fs::path(s.runs_dir));
        s.runs_refresh = false;
        if (s.runs_selected >= (int)runs_cache.size()) s.runs_selected = -1;
    }

    const float scale = g_dpi_scale;

    // ---- Folder input + Refresh ----
    card_begin("##runs_inputs");
    if (card_header(s, "##runs_inputs", "Run archive")) {
        labeled_path("##runs_dir", "Folder", s.runs_dir, sizeof(s.runs_dir), PickKind::Folder);
        if (ImGui::Button("Refresh", ImVec2(110.0f * scale, 0))) s.runs_refresh = true;
        ImGui::SameLine();
        ImGui::TextDisabled("(%zu run%s)", runs_cache.size(), runs_cache.size() == 1 ? "" : "s");
    }
    card_end();
    ImGui::Dummy(ImVec2(0, 6.0f * scale));

    // ---- Table ----
    card_begin("##runs_table");
    if (card_header(s, "##runs_table", "Past runs")) {
    if (runs_cache.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, rgb(140, 150, 170));
        ImGui::TextWrapped("No runs archived yet. Train a model on the Train tab and "
                           "it will be saved here automatically.");
        ImGui::PopStyleColor();
    } else {
        const ImGuiTableFlags tf =
            ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
            ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable | ImGuiTableFlags_Hideable;
        ImVec2 table_size(0, ImGui::GetTextLineHeightWithSpacing() * 14);
        if (ImGui::BeginTable("##runs_tbl", 7, tf, table_size)) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("When",     ImGuiTableColumnFlags_WidthFixed, 160.0f * scale);
            ImGui::TableSetupColumn("Test",     ImGuiTableColumnFlags_WidthFixed,  70.0f * scale);
            ImGui::TableSetupColumn("Drop",     ImGuiTableColumnFlags_WidthFixed,  60.0f * scale);
            ImGui::TableSetupColumn("Params",   ImGuiTableColumnFlags_WidthFixed,  70.0f * scale);
            ImGui::TableSetupColumn("Arch",     ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Epochs / LR", ImGuiTableColumnFlags_WidthFixed, 110.0f * scale);
            ImGui::TableSetupColumn("Note",     ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();

            for (int i = 0; i < (int)runs_cache.size(); i++) {
                const RunInfo& r = runs_cache[i];
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(i);
                bool sel = (s.runs_selected == i);
                if (ImGui::Selectable("##row", sel,
                                      ImGuiSelectableFlags_SpanAllColumns,
                                      ImVec2(0, ImGui::GetTextLineHeightWithSpacing()))) {
                    s.runs_selected = i;
                }
                ImGui::SameLine(0, 0);
                ImGui::TextUnformatted(r.timestamp.c_str());

                ImGui::TableSetColumnIndex(1);
                if (r.eval_total > 0) {
                    ImVec4 col = r.test_accuracy >= 0.85 ? rgb(110, 200, 130)
                              : r.test_accuracy >= 0.60 ? rgb(220, 200, 90)
                                                        : rgb(220, 110, 110);
                    ImGui::TextColored(col, "%.1f%%", 100.0 * r.test_accuracy);
                } else {
                    ImGui::TextDisabled("--");
                }
                ImGui::TableSetColumnIndex(2);
                if (r.eval_total > 0) ImGui::Text("%.0f%%", 100.0 * r.per_class_acc[3]);
                else                  ImGui::TextDisabled("--");
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%lld", r.params);
                ImGui::TableSetColumnIndex(4);
                ImGui::Text("conv %u/k%u/s%u + %ux%u", r.arch.conv_channels, r.arch.conv_kernel,
                            r.arch.conv_stride, r.arch.hidden_layers, r.arch.hidden_units);
                ImGui::TableSetColumnIndex(5);
                ImGui::Text("%d @ %.4f", r.epochs, r.learning_rate);
                ImGui::TableSetColumnIndex(6);
                ImGui::TextUnformatted(r.note.empty() ? "--" : r.note.c_str());
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }
    }  // end of if (card_header ... Past runs)
    card_end();

    // ---- Action buttons for the selected row ----
    if (s.runs_selected >= 0 && s.runs_selected < (int)runs_cache.size()) {
        const RunInfo& r = runs_cache[s.runs_selected];
        ImGui::Dummy(ImVec2(0, 4.0f * scale));
        if (ImGui::Button("Use as current model", ImVec2(220.0f * scale, 0))) {
            // Copy the selected run's model.json over Train's output path AND every
            // other tab's "input model" path, so the choice propagates everywhere.
            fs::path src = r.folder / "model.json";
            fs::path dst = fs::path(s.train_model_out);
            std::error_code ec;
            fs::create_directories(dst.parent_path(), ec); ec.clear();
            fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
            if (ec) {
                s.log.logf("Could not copy %s -> %s: %s\n",
                           src.string().c_str(), dst.string().c_str(), ec.message().c_str());
            } else {
                // Sync the Predict + Tune tab path fields so they all point at the same file.
                std::strncpy(s.predict_model,   dst.string().c_str(), sizeof(s.predict_model)   - 1);
                std::strncpy(s.tune_model_in,   dst.string().c_str(), sizeof(s.tune_model_in)   - 1);
                s.predict_model[sizeof(s.predict_model) - 1] = '\0';
                s.tune_model_in[sizeof(s.tune_model_in) - 1] = '\0';
                // Also sync the arch fields from this run so the Train tab's controls reflect it.
                s.arch_conv_channels = (int)r.arch.conv_channels;
                s.arch_conv_kernel   = (int)r.arch.conv_kernel;
                s.arch_conv_stride   = (int)r.arch.conv_stride;
                s.arch_hidden_layers = (int)r.arch.hidden_layers;
                s.arch_hidden_units  = (int)r.arch.hidden_units;
                s.log.logf("Loaded run %s -> %s  (predict + tune now point here)\n",
                           r.folder.filename().string().c_str(), dst.string().c_str());
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Delete run", ImVec2(140.0f * scale, 0))) {
            std::error_code ec;
            fs::remove_all(r.folder, ec);
            if (ec) s.log.logf("Could not delete %s: %s\n",
                               r.folder.string().c_str(), ec.message().c_str());
            else    s.log.logf("Deleted %s\n", r.folder.string().c_str());
            s.runs_refresh = true;
            s.runs_selected = -1;
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Selected: %s", r.folder.filename().string().c_str());
    }
}

// ============================================================================
// Dear ImGui Win32 + DX11 plumbing (adapted from the official example)
// ============================================================================
static ID3D11Device*           g_pd3dDevice           = nullptr;
static ID3D11DeviceContext*    g_pd3dDeviceContext    = nullptr;
static IDXGISwapChain*         g_pSwapChain           = nullptr;
static bool                    g_SwapChainOccluded    = false;
static UINT                    g_ResizeWidth = 0, g_ResizeHeight = 0;
static ID3D11RenderTargetView* g_mainRenderTargetView = nullptr;

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND, UINT, WPARAM, LPARAM);
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);

static void apply_modern_style() {
    ImGuiStyle& st = ImGui::GetStyle();

    // Geometry: tight enough that a typical 1180x820 window shows a whole tab
    // without scrolling, generous enough that rows don't feel cramped.
    st.WindowPadding     = ImVec2(18, 14);
    st.FramePadding      = ImVec2(10, 6);
    st.CellPadding       = ImVec2(8, 4);
    st.ItemSpacing       = ImVec2(10, 6);
    st.ItemInnerSpacing  = ImVec2(6, 4);
    st.IndentSpacing     = 20.0f;
    st.ScrollbarSize     = 10.0f;
    st.GrabMinSize       = 10.0f;

    st.WindowBorderSize  = 0.0f;
    st.FrameBorderSize   = 0.0f;
    st.ChildBorderSize   = 1.0f;
    st.PopupBorderSize   = 0.0f;
    st.TabBarBorderSize  = 0.0f;
    st.ButtonTextAlign   = ImVec2(0.5f, 0.5f);  // centered
    st.SelectableTextAlign = ImVec2(0.0f, 0.5f);

    st.WindowRounding    = 0.0f;
    st.ChildRounding     = 8.0f;
    st.FrameRounding     = 6.0f;
    st.PopupRounding     = 8.0f;
    st.ScrollbarRounding = 12.0f;
    st.GrabRounding      = 6.0f;
    st.TabRounding       = 8.0f;

    ImVec4* c = st.Colors;

    // Base surfaces
    c[ImGuiCol_WindowBg]            = rgb( 17,  20,  26);
    c[ImGuiCol_ChildBg]             = rgb( 21,  25,  32);
    c[ImGuiCol_PopupBg]             = rgb( 24,  28,  36, 0.97f);
    c[ImGuiCol_MenuBarBg]           = rgb( 17,  20,  26);
    c[ImGuiCol_ScrollbarBg]         = rgb( 17,  20,  26);

    // Frames
    c[ImGuiCol_FrameBg]             = rgb( 28,  33,  42);
    c[ImGuiCol_FrameBgHovered]      = rgb( 38,  46,  60);
    c[ImGuiCol_FrameBgActive]       = rgb( 48,  60,  80);

    // Borders / separators
    c[ImGuiCol_Border]              = rgb( 35,  40,  50);
    c[ImGuiCol_BorderShadow]        = rgb(  0,   0,   0, 0);
    c[ImGuiCol_Separator]           = rgb( 30,  35,  44);
    c[ImGuiCol_SeparatorHovered]    = rgb( 60,  85, 130);
    c[ImGuiCol_SeparatorActive]     = rgb( 80, 130, 200);

    // Text
    c[ImGuiCol_Text]                = rgb(225, 228, 234);
    c[ImGuiCol_TextDisabled]        = rgb(120, 128, 142);

    // Tabs
    c[ImGuiCol_Tab]                 = rgb( 21,  25,  32);
    c[ImGuiCol_TabHovered]          = rgb( 48,  80, 130);
    c[ImGuiCol_TabActive]           = rgb( 38,  74, 130);
    c[ImGuiCol_TabUnfocused]        = rgb( 17,  20,  26);
    c[ImGuiCol_TabUnfocusedActive]  = rgb( 28,  46,  74);

    // Buttons - subtle blue accent
    c[ImGuiCol_Button]              = rgb( 42,  82, 140);
    c[ImGuiCol_ButtonHovered]       = rgb( 60, 110, 180);
    c[ImGuiCol_ButtonActive]        = rgb( 80, 140, 220);

    // Header (selectable, tree)
    c[ImGuiCol_Header]              = rgb( 34,  43,  56);
    c[ImGuiCol_HeaderHovered]       = rgb( 48,  64,  86);
    c[ImGuiCol_HeaderActive]        = rgb( 60,  88, 130);

    // Sliders / check
    c[ImGuiCol_CheckMark]           = rgb(110, 175, 250);
    c[ImGuiCol_SliderGrab]          = rgb(110, 175, 250);
    c[ImGuiCol_SliderGrabActive]    = rgb(150, 200, 250);

    // Scrollbar
    c[ImGuiCol_ScrollbarGrab]        = rgb( 50,  60,  76);
    c[ImGuiCol_ScrollbarGrabHovered] = rgb( 70,  85, 110);
    c[ImGuiCol_ScrollbarGrabActive]  = rgb( 90, 120, 160);

    // Misc
    c[ImGuiCol_PlotHistogram]        = rgb(110, 175, 250);
    c[ImGuiCol_PlotHistogramHovered] = rgb(150, 200, 250);
}

// Draws the custom title bar (full window width) and advances the cursor to just
// below it. Handles min / max / close clicks. Must be called immediately after
// ImGui::Begin on the fullscreen workbench window.
static void draw_titlebar(AppState& app) {
    const float scale = g_dpi_scale;
    const float tb_h  = TITLEBAR_H_UNSCALED * scale;
    const float btn_w = WIN_BTN_W_UNSCALED  * scale;

    ImVec2 win_pos = ImGui::GetWindowPos();
    float  win_w   = ImGui::GetWindowWidth();
    ImDrawList* dl = ImGui::GetWindowDrawList();

    // Background fill
    ImVec2 tb_tl(win_pos.x, win_pos.y);
    ImVec2 tb_br(win_pos.x + win_w, win_pos.y + tb_h);
    dl->AddRectFilled(tb_tl, tb_br, IM_COL32(13, 16, 22, 255));
    // Hairline below
    dl->AddLine(ImVec2(tb_tl.x, tb_br.y - 1), ImVec2(tb_br.x, tb_br.y - 1),
                IM_COL32(38, 44, 56, 255), 1.0f);

    // ----- Title text (left) -----
    // Track the same side padding the content area uses (bumps up in narrow mode).
    const bool tb_narrow = win_w < NARROW_PX_UNSCALED * scale;
    const float tb_left_pad = (tb_narrow ? 56.0f : 36.0f) * scale;
    {
        if (g_font_h2) ImGui::PushFont(g_font_h2);
        float text_h = ImGui::GetTextLineHeight();
        ImGui::SetCursorScreenPos(ImVec2(tb_tl.x + tb_left_pad, tb_tl.y + (tb_h - text_h) * 0.5f));
        ImGui::TextUnformatted("Drop Detector");
        ImVec2 after = ImGui::GetItemRectMax();
        if (g_font_h2) ImGui::PopFont();

        // "Workbench" subtitle, baseline aligned
        ImGui::SameLine(0, 8 * scale);
        ImGui::SetCursorScreenPos(ImVec2(after.x + 8 * scale,
                                         tb_tl.y + (tb_h - ImGui::GetTextLineHeight()) * 0.5f));
        ImGui::TextDisabled("Workbench");
    }

    // ----- Status pill (right of title, left of window buttons) -----
    {
        const bool busy = app.busy.load();
        const char* label = busy ? "Working" : "Ready";
        const ImVec4 col = busy ? rgb(240, 180, 60) : rgb(80, 200, 120);
        const float pill_h = 22.0f * scale;
        const float dot_d  = 7.0f * scale;
        const float pad    = 10.0f * scale;
        const float label_w = ImGui::CalcTextSize(label).x;
        const float pill_w = label_w + pad * 2 + dot_d + 6 * scale;
        const float win_btns_total = btn_w * NUM_WIN_BTNS;
        ImVec2 pill_tl(tb_br.x - win_btns_total - pill_w - 12 * scale,
                       tb_tl.y + (tb_h - pill_h) * 0.5f);
        ImVec2 pill_br(pill_tl.x + pill_w, pill_tl.y + pill_h);
        auto mix = [](ImVec4 a, float k) {
            return IM_COL32((int)(a.x*255*k), (int)(a.y*255*k), (int)(a.z*255*k), 255);
        };
        dl->AddRectFilled(pill_tl, pill_br, mix(col, 0.18f), pill_h * 0.5f);
        dl->AddRect      (pill_tl, pill_br, mix(col, 0.55f), pill_h * 0.5f, 0, 1.0f);
        ImVec2 dot_c(pill_tl.x + pad + dot_d * 0.5f, pill_tl.y + pill_h * 0.5f);
        dl->AddCircleFilled(dot_c, dot_d * 0.5f, mix(col, 1.0f));
        ImVec2 text_p(dot_c.x + dot_d * 0.5f + 6 * scale,
                      pill_tl.y + (pill_h - ImGui::GetTextLineHeight()) * 0.5f);
        dl->AddText(text_p, mix(col, 1.0f), label);
    }

    // ----- Window control buttons (right edge: minimize / maximize / close) -----
    auto window_btn = [&](const char* id, float x_right_offset, ImU32 hover, bool is_close,
                          std::function<void(ImDrawList*, ImVec2)> icon) -> bool {
        ImVec2 pos(tb_br.x - x_right_offset, tb_tl.y);
        ImVec2 size(btn_w, tb_h);
        ImGui::SetCursorScreenPos(pos);
        ImGui::PushID(id);
        bool clicked = ImGui::InvisibleButton("##b", size);
        bool hovered = ImGui::IsItemHovered();
        ImGui::PopID();
        if (hovered) dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), hover);
        icon(dl, ImVec2(pos.x + size.x * 0.5f, pos.y + size.y * 0.5f));
        (void)is_close;
        return clicked;
    };

    const ImU32 icon_col   = IM_COL32(225, 228, 234, 255);
    const ImU32 hover_neut = IM_COL32(45, 52, 65, 255);
    const ImU32 hover_red  = IM_COL32(232, 17, 35, 255);
    const float thick      = 1.0f * scale;
    const bool  maximized  = g_main_hwnd && ::IsZoomed(g_main_hwnd);

    if (window_btn("##min_btn", btn_w * 3, hover_neut, false,
        [&](ImDrawList* d, ImVec2 c) {
            float r = 5.5f * scale;
            d->AddLine(ImVec2(c.x - r, c.y + 0.5f), ImVec2(c.x + r, c.y + 0.5f), icon_col, thick);
        })) {
        if (g_main_hwnd) ::ShowWindow(g_main_hwnd, SW_MINIMIZE);
    }
    if (window_btn("##max_btn", btn_w * 2, hover_neut, false,
        [&](ImDrawList* d, ImVec2 c) {
            float r = 4.5f * scale;
            if (maximized) {
                float o = 1.8f * scale;
                d->AddRect(ImVec2(c.x - r + o, c.y - r - o), ImVec2(c.x + r + o, c.y + r - o),
                           icon_col, 0, 0, thick);
                d->AddRectFilled(ImVec2(c.x - r - o, c.y - r + o), ImVec2(c.x + r - o, c.y + r + o),
                                 IM_COL32(13, 16, 22, 255));
                d->AddRect(ImVec2(c.x - r - o, c.y - r + o), ImVec2(c.x + r - o, c.y + r + o),
                           icon_col, 0, 0, thick);
            } else {
                d->AddRect(ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r),
                           icon_col, 0, 0, thick);
            }
        })) {
        if (g_main_hwnd) ::ShowWindow(g_main_hwnd, maximized ? SW_RESTORE : SW_MAXIMIZE);
    }
    if (window_btn("##close_btn", btn_w * 1, hover_red, true,
        [&](ImDrawList* d, ImVec2 c) {
            float r = 5.0f * scale;
            d->AddLine(ImVec2(c.x - r, c.y - r), ImVec2(c.x + r, c.y + r), icon_col, thick);
            d->AddLine(ImVec2(c.x - r, c.y + r), ImVec2(c.x + r, c.y - r), icon_col, thick);
        })) {
        if (g_main_hwnd) ::PostMessageW(g_main_hwnd, WM_CLOSE, 0, 0);
    }

    // Reserve the row and position the cursor for the next widget.
    ImGui::SetCursorScreenPos(ImVec2(win_pos.x, win_pos.y + tb_h));
}

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
    AppState app;

    // COM init -- IFileOpenDialog (folder picker) needs apartment-threaded COM on the UI thread.
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

    // Per-monitor DPI awareness -- must be set BEFORE creating any window so
    // Windows doesn't bitmap-scale our backbuffer for HiDPI displays.
    ImGui_ImplWin32_EnableDpiAwareness();

    WNDCLASSEXW wc = { sizeof(wc), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(nullptr),
                       nullptr, nullptr, nullptr, nullptr, L"DropDetectWorkbench", nullptr };
    ::RegisterClassExW(&wc);
    HWND hwnd = ::CreateWindowW(wc.lpszClassName, L"Drop Detector",
                                WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 1280, 820,
                                nullptr, nullptr, wc.hInstance, nullptr);
    g_main_hwnd = hwnd;

    // Tell DWM to draw its shadow + Win11 rounded corners around our frameless window.
    // A nonzero margins value triggers the shadow; we pick (1,1,1,1) so it's invisible.
    {
        MARGINS m{ 1, 1, 1, 1 };
        ::DwmExtendFrameIntoClientArea(hwnd, &m);
        BOOL dark = TRUE;
        ::DwmSetWindowAttribute(hwnd, /*DWMWA_USE_IMMERSIVE_DARK_MODE=*/20, &dark, sizeof(dark));
        // Force WM_NCCALCSIZE to fire so our frameless layout takes effect immediately.
        ::SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                       SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED);
    }

    if (!CreateDeviceD3D(hwnd)) {
        CleanupDeviceD3D();
        ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 1;
    }
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // Get the real DPI scale for this monitor. 1.0 = 96 DPI, 1.5 = 144 DPI, etc.
    g_dpi_scale = ImGui_ImplWin32_GetDpiScaleForHwnd(hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr;  // don't litter the working dir with imgui.ini
    ImGui::StyleColorsDark();
    apply_modern_style();
    ImGui::GetStyle().ScaleAllSizes(g_dpi_scale);

    // Load real Windows fonts at the monitor's physical pixel size.
    {
        ImFontConfig cfg;
        cfg.OversampleH = 3;
        cfg.OversampleV = 2;
        cfg.PixelSnapH  = false;
        const char* ui_path   = "C:\\Windows\\Fonts\\segoeui.ttf";
        const char* ui_bold   = "C:\\Windows\\Fonts\\segoeuib.ttf";
        const char* ui_semib  = "C:\\Windows\\Fonts\\seguisb.ttf";
        const char* mono_path = "C:\\Windows\\Fonts\\consola.ttf";
        if (fs::exists(ui_path))   g_font_ui   = io.Fonts->AddFontFromFileTTF(ui_path,   16.0f * g_dpi_scale, &cfg);
        if (fs::exists(ui_bold))   g_font_h1   = io.Fonts->AddFontFromFileTTF(ui_bold,   26.0f * g_dpi_scale, &cfg);
        if (fs::exists(ui_semib))  g_font_h2   = io.Fonts->AddFontFromFileTTF(ui_semib,  17.0f * g_dpi_scale, &cfg);
        else if (fs::exists(ui_bold)) g_font_h2 = io.Fonts->AddFontFromFileTTF(ui_bold,  17.0f * g_dpi_scale, &cfg);
        if (fs::exists(mono_path)) g_font_mono = io.Fonts->AddFontFromFileTTF(mono_path, 13.0f * g_dpi_scale, &cfg);
        if (g_font_ui) io.FontDefault = g_font_ui;
        io.Fonts->Build();
    }

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    bool done = false;
    while (!done) {
        MSG msg;
        while (::PeekMessage(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT) done = true;
        }
        if (done) break;

        if (g_SwapChainOccluded && g_pSwapChain->Present(0, DXGI_PRESENT_TEST) == DXGI_STATUS_OCCLUDED) {
            ::Sleep(10); continue;
        }
        g_SwapChainOccluded = false;

        if (g_ResizeWidth != 0 && g_ResizeHeight != 0) {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, g_ResizeWidth, g_ResizeHeight, DXGI_FORMAT_UNKNOWN, 0);
            g_ResizeWidth = g_ResizeHeight = 0;
            CreateRenderTarget();
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // Fullscreen workbench window -- zero padding so we can draw a flush title bar.
        const ImGuiViewport* vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->WorkPos);
        ImGui::SetNextWindowSize(vp->WorkSize);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
        ImGui::Begin("##workbench", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar);
        ImGui::PopStyleVar(2);

        // Custom title bar (replaces the OS title bar). Draws + handles min/max/close.
        draw_titlebar(app);

        // Content area below the title bar -- side padding only, modest top/bottom.
        // When the window is in the narrow/minimized layout, push the side padding
        // up so the single-column content doesn't visually hug the window edges.
        const bool content_narrow = vp->WorkSize.x < NARROW_PX_UNSCALED * g_dpi_scale;
        const float side_pad = (content_narrow ? 56.0f : 36.0f) * g_dpi_scale;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,
                            ImVec2(side_pad, 14.0f * g_dpi_scale));
        ImGui::BeginChild("##content", ImVec2(0, 0), ImGuiChildFlags_None,
                          ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        {
            // Burger menu (replaces the tab bar). Updates app.active_tab on selection.
            draw_burger_nav(app);
            ImGui::Dummy(ImVec2(0, 10.0f * g_dpi_scale));

            switch (app.active_tab) {
                case Tab::Build:   render_build_tab(app);   break;
                case Tab::Train:   render_train_tab(app);   break;
                case Tab::Predict: render_predict_tab(app); break;
                case Tab::Tune:    render_tune_tab(app);    break;
                case Tab::Runs:    render_runs_tab(app);    break;
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
        ImGui::End();

        ImGui::Render();
        const float clear[4] = { 0.06f, 0.06f, 0.07f, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        HRESULT hr = g_pSwapChain->Present(1, 0);
        g_SwapChainOccluded = (hr == DXGI_STATUS_OCCLUDED);
    }

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClassW(wc.lpszClassName, wc.hInstance);
    CoUninitialize();
    return 0;
}

bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    UINT create_flags = 0;
    D3D_FEATURE_LEVEL feature_level;
    const D3D_FEATURE_LEVEL feature_levels[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    HRESULT res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, create_flags,
        feature_levels, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice,
        &feature_level, &g_pd3dDeviceContext);
    if (res == DXGI_ERROR_UNSUPPORTED)
        res = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, create_flags,
            feature_levels, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice,
            &feature_level, &g_pd3dDeviceContext);
    if (res != S_OK) return false;
    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain)        { g_pSwapChain->Release();        g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice)        { g_pd3dDevice->Release();        g_pd3dDevice = nullptr; }
}

void CreateRenderTarget() {
    ID3D11Texture2D* pBackBuffer = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget() {
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) return true;
    switch (msg) {
    case WM_NCCALCSIZE: {
        // Frameless window: extend the client area over what would normally be the
        // window frame so we can draw our own title bar.
        if (wParam == TRUE) {
            NCCALCSIZE_PARAMS* p = reinterpret_cast<NCCALCSIZE_PARAMS*>(lParam);
            if (::IsZoomed(hWnd)) {
                // When maximized, Windows pushes the window past the screen edges by
                // the frame size; pull the client area back in so nothing is clipped.
                UINT dpi = ::GetDpiForWindow(hWnd);
                int frame_x = ::GetSystemMetricsForDpi(SM_CXFRAME, dpi)
                            + ::GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
                int frame_y = ::GetSystemMetricsForDpi(SM_CYFRAME, dpi)
                            + ::GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
                p->rgrc[0].top    += frame_y;
                p->rgrc[0].bottom -= frame_y;
                p->rgrc[0].left   += frame_x;
                p->rgrc[0].right  -= frame_x;
            }
            // When not maximized: leave rect unmodified -> client area = full window.
            return 0;
        }
        break;
    }
    case WM_NCHITTEST: {
        // First let DWM decide for system gestures (snap layouts on Win11, etc.).
        LRESULT dwm_hit = 0;
        if (::DwmDefWindowProc(hWnd, msg, wParam, lParam, &dwm_hit) && dwm_hit) return dwm_hit;

        POINT pt{ GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
        RECT r; ::GetWindowRect(hWnd, &r);
        UINT dpi  = ::GetDpiForWindow(hWnd);
        float scale = dpi ? dpi / 96.0f : 1.0f;
        int border_x = ::GetSystemMetricsForDpi(SM_CXFRAME, dpi)
                     + ::GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
        int border_y = ::GetSystemMetricsForDpi(SM_CYFRAME, dpi)
                     + ::GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi);
        int title_h  = (int)(TITLEBAR_H_UNSCALED * scale);
        int btns_w   = (int)(WIN_BTN_W_UNSCALED * scale) * NUM_WIN_BTNS;
        bool maximized = ::IsZoomed(hWnd);

        // Resize edges (skip when maximized)
        if (!maximized) {
            bool L = pt.x < r.left   + border_x;
            bool R = pt.x >= r.right  - border_x;
            bool T = pt.y < r.top    + border_y;
            bool B = pt.y >= r.bottom - border_y;
            if (T && L) return HTTOPLEFT;
            if (T && R) return HTTOPRIGHT;
            if (B && L) return HTBOTTOMLEFT;
            if (B && R) return HTBOTTOMRIGHT;
            if (T)      return HTTOP;
            if (B)      return HTBOTTOM;
            if (L)      return HTLEFT;
            if (R)      return HTRIGHT;
        }

        // Title bar strip (top of window). Exclude the 3-button area on the right
        // so ImGui receives those clicks.
        if (pt.y < r.top + title_h) {
            if (pt.x >= r.right - btns_w) return HTCLIENT;
            return HTCAPTION;
        }
        return HTCLIENT;
    }
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED) return 0;
        g_ResizeWidth  = (UINT)LOWORD(lParam);
        g_ResizeHeight = (UINT)HIWORD(lParam);
        return 0;
    case WM_GETMINMAXINFO: {
        LPMINMAXINFO mm = reinterpret_cast<LPMINMAXINFO>(lParam);
        UINT dpi = ::GetDpiForWindow(hWnd);
        float scale = dpi ? dpi / 96.0f : 1.0f;
        mm->ptMinTrackSize.x = (LONG)(960 * scale);
        mm->ptMinTrackSize.y = (LONG)(640 * scale);
        return 0;
    }
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0); return 0;
    }
    return ::DefWindowProcW(hWnd, msg, wParam, lParam);
}
