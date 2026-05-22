#define BT_MAX_LAYERS 32
#define BT_L_MAX_DIMS 3
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#define INT_MIN  (-2147483648)
#define INT_MAX  (2147483647)
#define LONG_MIN  (-9223372036854775808L)
#define LONG_MAX  (9223372036854775807L)
#define FLOAT_MIN  (1.175494351e-38f)
#define FLOAT_MAX  (3.402823466e+38f)
#define FLOAT_LOWEST (-3.402823466e+38f)
#define DOUBLE_MIN  (2.2250738585072014e-308)
#define DOUBLE_MAX  (1.7976931348623157e+308)
#define DOUBLE_LOWEST (-1.7976931348623157e+308)

using float32_t = float;
using float64_t = double;

namespace BeanTensor::lite::Tensors {
    struct Tensor;
}

namespace BeanTensor::lite {
    using DataPtr = std::variant<float32_t*, float64_t*, int32_t*, int64_t*>;
    enum Precision {
        FP32,
        FP64,
        INT32,
        INT64,
    };
}

namespace BeanTensor::lite::detail {

    template<typename Fn>
    double run_bench_tops(Fn fn, const int iterations, const size_t N) {
        for (int i = 0; i < 10; i++) fn();
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) fn();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double elapsed   = std::chrono::duration<double>(t1 - t0).count();
        const double total_ops = static_cast<double>(N) * iterations;
        const double tops      = total_ops / elapsed / 1e12;
        return tops;
    }

    size_t get_precision_length(const Precision& p) {
        switch (p) {
            case FP32: return sizeof(float);
            case FP64: return sizeof(double);
            case INT32: return sizeof(int32_t);
            case INT64: return sizeof(int64_t);
        }
        throw std::runtime_error("Invalid precision");
    }
}

namespace BeanTensor::lite::Tensors {
    struct Tensor {
        uint32_t _dims[BT_L_MAX_DIMS]{0};
        uint8_t _dims_l;
        DataPtr _data;
        Precision _precision;

        [[nodiscard]] size_t numel() const {
            size_t n = 1;
            for (uint32_t i = 0; i < this->_dims_l; i++) n *= this->_dims[i];
            return n;
        }



        void set(const std::initializer_list<size_t> idx, const double value) {
            if (idx.size() != this->_dims_l)
                throw std::invalid_argument("Tensor index size does not match tensor dimensions");
            size_t offset = 0;
            for (size_t i = 0; i < this->_dims_l; i++) {
                if (idx.begin()[i] < this->_dims[i]) {
                    offset *= this->_dims[i];
                    offset += idx.begin()[i];
                    continue;
                }
                throw std::invalid_argument("Tensor index size goes out of bounds.");
            }
            Precision dtype = this->_precision;
            std::visit([offset, value, dtype](auto* ptr) {
                switch (dtype) {
                    case FP32:  ptr[offset] = static_cast<float32_t>(value); break;
                    case FP64:  ptr[offset] = static_cast<float64_t>(value); break;
                    case INT32: ptr[offset] = static_cast<int32_t>(value);   break;
                    case INT64: ptr[offset] = static_cast<int64_t>(value);   break;
                }
            }, this->_data);
        }

        void set_random(const double min = 0, const double max = 1, const size_t seed = 42) {
            if (min >= max) throw std::invalid_argument("Min must be less than max");
            switch (this->_precision) {
                case FP32:  if (min > FLOAT_MAX  || max < FLOAT_LOWEST)  throw std::invalid_argument("Min or max out of range for FP32");  break;
                case FP64:  if (min > DOUBLE_MAX || max < DOUBLE_LOWEST) throw std::invalid_argument("Min or max out of range for FP64");  break;
                case INT32: if (min > INT_MAX    || max < INT_MIN)    throw std::invalid_argument("Min or max out of range for INT32"); break;
                case INT64: if (static_cast<int64_t>(min) > LONG_MAX || max < LONG_MIN) throw std::invalid_argument("Min or max out of range for INT64"); break;
            }
            std::mt19937 rng(seed);
            std::visit([&](auto* ptr) {
                using T = std::remove_pointer_t<decltype(ptr)>;
                if constexpr (std::is_floating_point_v<T>) {
                    std::uniform_real_distribution<double> dist(min, max);
                    for (size_t i = 0; i < this->numel(); i++) ptr[i] = static_cast<T>(dist(rng));
                } else {
                    std::uniform_int_distribution<int64_t> dist(static_cast<int64_t>(min), static_cast<int64_t>(max));
                    for (size_t i = 0; i < this->numel(); i++) ptr[i] = static_cast<T>(dist(rng));
                }
            }, this->_data);
        }

        void all_zeros() {
            std::visit([&](auto* ptr) {
                for (size_t i = 0; i < this->numel(); i++) {
                    switch (this->_precision) {
                        case FP32:  ptr[i] = static_cast<float32_t>(0); break;
                        case FP64:  ptr[i] = static_cast<float64_t>(0); break;
                        case INT32: ptr[i] = static_cast<int32_t>(0);   break;
                        case INT64: ptr[i] = static_cast<int64_t>(0);   break;
                    }
                }
            }, this->_data);
        }

        void all_ones() {
            std::visit([&](auto* ptr) {
                for (size_t i = 0; i < this->numel(); i++) {
                    switch (this->_precision) {
                        case FP32:  ptr[i] = static_cast<float32_t>(1); break;
                        case FP64:  ptr[i] = static_cast<float64_t>(1); break;
                        case INT32: ptr[i] = static_cast<int32_t>(1);   break;
                        case INT64: ptr[i] = static_cast<int64_t>(1);   break;
                    }
                }
            }, this->_data);
        }

        ~Tensor() {
            std::visit([](const auto* ptr) { delete[] ptr; }, this->_data);
        }

        Tensor(const Tensor& other) : _dims_l(other._dims_l), _precision(other._precision) {
            std::copy(other._dims, other._dims + BT_L_MAX_DIMS, _dims);
            std::visit([&](const auto* src) {
                using T = std::remove_const_t<std::remove_pointer_t<decltype(src)>>;
                auto* dst = new T[numel()];
                std::copy(src, src + numel(), dst);
                _data = dst;
            }, other._data);
        }

        Tensor(Tensor&& other) noexcept : _dims_l(other._dims_l), _data(other._data), _precision(other._precision) {
            std::copy(other._dims, other._dims + BT_L_MAX_DIMS, _dims);
            std::visit([](auto*& ptr) { ptr = nullptr; }, other._data);
        }

        Tensor& operator=(const Tensor& other) {
            if (this == &other) return *this;
            std::visit([](const auto* ptr) { delete[] ptr; }, _data);
            _dims_l = other._dims_l;
            _precision = other._precision;
            std::copy(other._dims, other._dims + BT_L_MAX_DIMS, _dims);
            std::visit([&](const auto* src) {
                using T = std::remove_const_t<std::remove_pointer_t<decltype(src)>>;
                auto* dst = new T[numel()];
                std::copy(src, src + numel(), dst);
                _data = dst;
            }, other._data);
            return *this;
        }

        Tensor& operator=(Tensor&& other) noexcept {
            if (this == &other) return *this;
            std::visit([](const auto* ptr) { delete[] ptr; }, _data);
            _dims_l = other._dims_l;
            _precision = other._precision;
            _data = other._data;
            std::copy(other._dims, other._dims + BT_L_MAX_DIMS, _dims);
            std::visit([](auto*& ptr) { ptr = nullptr; }, other._data);
            return *this;
        }

        Tensor(const std::vector<size_t>& dims, const Precision& precision) {
            if (dims.size() > BT_L_MAX_DIMS)  throw std::invalid_argument("Tensor dimension exceeds maximum supported dimensions");
            if (dims.empty())                  throw std::invalid_argument("Tensor must have at least one dimension");
            if (precision != FP32 && precision != FP64 && precision != INT32 && precision != INT64)
                throw std::invalid_argument("Invalid precision");
            this->_dims_l = dims.size();
            for (size_t i = 0; i < this->_dims_l; i++) this->_dims[i] = dims[i];
            this->_precision = precision;
            switch (this->_precision) {
                case FP32:  this->_data = new float[this->numel()];    break;
                case FP64:  this->_data = new double[this->numel()];   break;
                case INT32: this->_data = new int32_t[this->numel()];  break;
                case INT64: this->_data = new int64_t[this->numel()];  break;
            }
        }

        Tensor(const std::initializer_list<size_t>& dims, const Precision& precision) {
            if (dims.size() > BT_L_MAX_DIMS) throw std::invalid_argument("Tensor dimension exceeds maximum supported dimensions");
            if (dims.size() == 0)            throw std::invalid_argument("Tensor must have at least one dimension");
            if (precision != FP32 && precision != FP64 && precision != INT32 && precision != INT64)
                throw std::invalid_argument("Invalid precision");
            this->_dims_l = dims.size();
            for (size_t i = 0; i < this->_dims_l; i++) this->_dims[i] = dims.begin()[i];
            this->_precision = precision;
            switch (this->_precision) {
                case FP32:  this->_data = new float[this->numel()];    break;
                case FP64:  this->_data = new double[this->numel()];   break;
                case INT32: this->_data = new int32_t[this->numel()];  break;
                case INT64: this->_data = new int64_t[this->numel()];  break;
            }
        }

        static void _print_recursive(std::ostringstream& os, const Tensor& tensor, size_t dim, size_t offset) {
            os << "[";
            if (dim == tensor._dims_l - 1) {
                for (size_t i = 0; i < tensor._dims[dim]; i++) {
                    os << std::visit([idx = offset + i](const auto* ptr) -> double {
                        return static_cast<double>(ptr[idx]);
                    }, tensor._data);
                    if (i != tensor._dims[dim] - 1) os << ", ";
                }
            } else {
                size_t stride = 1;
                for (size_t d = dim + 1; d < tensor._dims_l; d++) stride *= tensor._dims[d];
                for (size_t i = 0; i < tensor._dims[dim]; i++) {
                    _print_recursive(os, tensor, dim + 1, offset + i * stride);
                    if (i != tensor._dims[dim] - 1) os << ", ";
                }
            }
            os << "]";
        }

        static std::string shape_to_string(const Tensor& tensor) {
            std::ostringstream os;
            _print_recursive(os, tensor, 0, 0);
            return os.str();
        }

        [[nodiscard]] double at(const std::initializer_list<size_t> idx) const {
            if (idx.size() != this->_dims_l)
                throw std::invalid_argument("Tensor index size does not match tensor dimensions");
            size_t offset = 0;
            for (size_t i = 0; i < this->_dims_l; i++) {
                if (idx.begin()[i] < this->_dims[i]) {
                    offset *= this->_dims[i];
                    offset += idx.begin()[i];
                    continue;
                }
                throw std::invalid_argument("Tensor index size goes out of bounds.");
            }
            return std::visit([offset](const auto* ptr) -> double {
                return static_cast<double>(ptr[offset]);
            }, this->_data);
        }
    };
}

namespace BeanTensor::lite::Loss {
    struct MSE {
        static double forward(const Tensors::Tensor& pred, const Tensors::Tensor& target) {
            if (pred.numel() != target.numel()) throw std::invalid_argument("Shape mismatch");
            double loss = 0.0;
            for (size_t i = 0; i < pred.numel(); i++) {
                double d = pred.at({i}) - target.at({i});
                loss += d * d;
            }
            return loss / static_cast<double>(pred.numel());
        }
        static Tensors::Tensor backward(const Tensors::Tensor& pred, const Tensors::Tensor& target, const Precision& precision) {
            auto grad = Tensors::Tensor({pred.numel()}, precision);
            double scale = 2.0 / static_cast<double>(pred.numel());
            for (size_t i = 0; i < pred.numel(); i++)
                grad.set({i}, scale * (pred.at({i}) - target.at({i})));
            return grad;
        }
    };
    struct CrossEntropy {
        static double forward(const Tensors::Tensor& pred, const Tensors::Tensor& target) {
            double loss = 0.0;
            for (size_t i = 0; i < pred.numel(); i++)
                loss -= target.at({i}) * std::log(pred.at({i}) + 1e-9);
            return loss;
        }
        static Tensors::Tensor backward(const Tensors::Tensor& pred, const Tensors::Tensor& target, const Precision& precision) {
            auto grad = Tensors::Tensor({pred.numel()}, precision);
            for (size_t i = 0; i < pred.numel(); i++)
                grad.set({i}, -target.at({i}) / (pred.at({i}) + 1e-9));
            return grad;
        }
    };
}

namespace BeanTensor::lite::Layers {
    struct Layer {
        virtual ~Layer() = default;
        [[nodiscard]] virtual Tensors::Tensor forward(const Tensors::Tensor& input) = 0;
        [[nodiscard]] virtual Tensors::Tensor backward(const Tensors::Tensor& grad_output, float32_t lr) = 0;
        [[nodiscard]] virtual std::string name() const = 0;
        [[nodiscard]] virtual size_t param_bytes() const { return 0; }
        std::optional<Tensors::Tensor> _last_input;
    };
    struct ReLU : Layer {
        Tensors::Tensor forward(const Tensors::Tensor& input) override {
            if (input._dims_l != 1) throw std::invalid_argument("ReLU expects 1D tensor");
            _last_input = input;
            auto output = Tensors::Tensor(input);
            for (size_t i = 0; i < output.numel(); i++) {
                if (output.at({i}) < 0.0) output.set({i}, 0.0);
            }
            return output;
        }

        Tensors::Tensor backward(const Tensors::Tensor& grad_output, float32_t lr) override {
            if (!_last_input) throw std::runtime_error("forward() must be called before backward()");
            auto d_input = Tensors::Tensor(*_last_input);
            for (size_t i = 0; i < d_input.numel(); i++) {
                d_input.set({i}, _last_input->at({i}) < 0.0 ? 0.0 : grad_output.at({i}));
            }
            return d_input;
        }

        [[nodiscard]] std::string name() const override { return "ReLU"; }
    };

    struct GlobalAveragePooling1D : Layer {
        Precision _precision;
        explicit GlobalAveragePooling1D(const Precision& precision) : _precision(precision) {}

        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& input) override {
            if (input._dims_l != 2) throw std::invalid_argument("Input Violation - Expected a two dimensional tensor");
            _last_input = input;
            const uint32_t channels = input._dims[0];
            const uint32_t length   = input._dims[1];
            auto output = Tensors::Tensor({channels}, _precision);
            output.all_zeros();
            for (size_t ch = 0; ch < channels; ch++) {
                double acc = 0.0;
                for (size_t pos = 0; pos < length; pos++) acc += input.at({ch, pos});
                output.set({ch}, acc / length);
            }
            return output;
        }

        [[nodiscard]] Tensors::Tensor backward(const Tensors::Tensor& grad_output, float32_t lr) override {
            if (!_last_input) throw std::runtime_error("forward() must be called before backward()");
            const auto& input = *_last_input;
            const uint32_t channels = input._dims[0];
            const uint32_t length   = input._dims[1];
            const double scale      = 1.0 / static_cast<double>(length);
            auto d_input = Tensors::Tensor({channels, length}, _precision);
            d_input.all_zeros();
            for (size_t ch = 0; ch < channels; ch++)
                for (size_t pos = 0; pos < length; pos++)
                    d_input.set({ch, pos}, grad_output.at({ch}) * scale);
            return d_input;
        }

        [[nodiscard]] std::string name() const override { return "GlobalAveragePooling1D"; }
    };

    struct Dense : Layer {
        uint32_t _in_features;
        uint32_t _out_features;
        Tensors::Tensor _weight;
        Tensors::Tensor _bias;
        Precision _precision;

        explicit Dense(const uint32_t in_features, const uint32_t out_features, const Precision& precision) :
            _weight({out_features, in_features}, precision),
            _bias({out_features}, precision) {
            this->_in_features  = in_features;
            this->_out_features = out_features;
            this->_precision    = precision;
            // Xavier uniform: keeps activation variance stable across layers
            const double limit = std::sqrt(6.0 / (in_features + out_features));
            this->_weight.set_random(-limit, limit);
            this->_bias.all_zeros();
        }

        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& input) override {
            if (input._dims_l != 1) throw std::invalid_argument("Input Violation - Expected a one dimensional tensor");
            _last_input = input;
            auto output = Tensors::Tensor({_out_features}, _precision);
            output.all_zeros();
            for (size_t o = 0; o < _out_features; o++) {
                double acc = 0.0;
                for (size_t i = 0; i < _in_features; i++) acc += _weight.at({o, i}) * input.at({i});
                output.set({o}, acc + _bias.at({o}));
            }
            return output;
        }

        [[nodiscard]] Tensors::Tensor backward(const Tensors::Tensor& grad_output, float32_t lr) override {
            if (!_last_input) throw std::runtime_error("forward() must be called before backward()");
            const auto& input = *_last_input;
            auto d_input = Tensors::Tensor({_in_features}, _precision);
            d_input.all_zeros();
            for (size_t i = 0; i < _in_features; i++) {
                double acc = 0.0;
                for (size_t o = 0; o < _out_features; o++) acc += _weight.at({o, i}) * grad_output.at({o});
                d_input.set({i}, acc);
            }
            for (size_t o = 0; o < _out_features; o++) {
                for (size_t i = 0; i < _in_features; i++) {
                    _weight.set({o, i}, _weight.at({o, i}) - lr * grad_output.at({o}) * input.at({i}));
                }
                _bias.set({o}, _bias.at({o}) - lr * grad_output.at({o}));
            }
            return d_input;
        }

        [[nodiscard]] std::string name() const override { return "Dense"; }
        [[nodiscard]] size_t param_bytes() const override {
            return std::visit([](const auto* ptr) {
                return sizeof(*ptr);
            }, _weight._data) * (_weight.numel() + _bias.numel());
        }
    };

    struct Softmax : Layer {
        Precision _precision;
        explicit Softmax(const Precision& precision) : _precision(precision) {}

        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& input) override {
            if (input._dims_l != 1) throw std::invalid_argument("Input Violation - Expected a one dimensional tensor");
            _last_input = input;
            auto output = Tensors::Tensor({input._dims[0]}, _precision);
            double max_val = input.at({0});
            for (size_t i = 1; i < input.numel(); i++) {
                if (input.at({i}) > max_val) max_val = input.at({i});
            }
            double sum = 0.0;
            for (size_t i = 0; i < input.numel(); i++) {
                output.set({i}, std::exp(input.at({i}) - max_val));
                sum += output.at({i});
            }
            for (size_t i = 0; i < output.numel(); i++) {
                output.set({i}, output.at({i}) / sum);
            }
            return output;
        }

        [[nodiscard]] Tensors::Tensor backward(const Tensors::Tensor& grad_output, float32_t lr) override {
            if (!_last_input) throw std::runtime_error("forward() must be called before backward()");
            auto softmax_out = forward(*_last_input);
            const size_t n = softmax_out.numel();
            auto d_input = Tensors::Tensor({n}, _precision);
            double dot = 0.0;
            for (size_t j = 0; j < n; j++) dot += softmax_out.at({j}) * grad_output.at({j});
            for (size_t i = 0; i < n; i++) {
                d_input.set({i}, softmax_out.at({i}) * (grad_output.at({i}) - dot));
            }
            return d_input;
        }

        [[nodiscard]] std::string name() const override { return "Softmax"; }
    };

    struct Sequential {
        std::vector<std::unique_ptr<Layer>> _layers;
        Precision _precision;

        Sequential(const Sequential&) = delete;
        Sequential& operator=(const Sequential&) = delete;
        Sequential(Sequential&&) = default;
        Sequential& operator=(Sequential&&) = default;

        explicit Sequential(const Precision& precision) : _precision(precision) {}

        template<typename T, typename... Args>
        Sequential& add(Args&&... args) {
            _layers.push_back(std::make_unique<T>(std::forward<Args>(args)...));
            return *this;
        }

        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& input) {
            Tensors::Tensor x = input;
            for (auto& layer : _layers) x = layer->forward(x);
            return x;
        }

        Tensors::Tensor backward(const Tensors::Tensor& grad, float32_t lr) {
            Tensors::Tensor g = grad;
            for (int i = static_cast<int>(_layers.size()) - 1; i >= 0; i--)
                g = _layers[i]->backward(g, lr);
            return g;
        }

        [[nodiscard]] size_t get_bytes() const {
            size_t total = 0;
            for (const auto& layer : _layers) total += layer->param_bytes();
            return total;
        }

        void summary() const {
            printf("%-5s %-20s\n", "Idx", "Layer");
            printf("---------------------------\n");
            for (size_t i = 0; i < _layers.size(); i++)
                printf("[%zu] %s\n", i, _layers[i]->name().c_str());
        }
    };
}

namespace BeanTensor::lite::DatasetOpener {
    struct Dataset {
        int32_t N{}, T{}, F{};
        std::vector<float> X;   // shape [N][T][F], row-major
        std::vector<int>   y;   // shape [N]
    };

    Dataset load_bin(const char* path) {
        FILE* f = fopen(path, "rb");
        Dataset d;
        fread(&d.N, sizeof(int), 1, f);
        fread(&d.T, sizeof(int), 1, f);
        fread(&d.F, sizeof(int), 1, f);
        d.X.resize(d.N * d.T * d.F);
        d.y.resize(d.N);
        fread(d.X.data(), sizeof(float), d.X.size(), f);
        fread(d.y.data(), sizeof(int),   d.y.size(), f);
        fclose(f);
        return d;
    }

}
using namespace BeanTensor::lite;

int main() {
    namespace fs = std::filesystem;
    const auto root = fs::canonical("/proc/self/exe").parent_path().parent_path();

    printf("Loading dataset...\n");
    auto [N, T, F, X, y] = DatasetOpener::load_bin((root / "TRAIN.bin").string().c_str());
    auto test  = DatasetOpener::load_bin((root / "TEST.bin").string().c_str());

    constexpr int NUM_CLASSES = 4;
    const int CHANNELS    = F;  // 4 features: a_mag, g_mag, jerk_a, jerk_g
    const int TIMESTEPS   = T;  // 249 timesteps

    printf("Train: %d  Test: %d  T=%d  F=%d  Classes=%d\n\n",
           N, test.N, TIMESTEPS, CHANNELS, NUM_CLASSES);

    Layers::Sequential model(FP32);
    model.add<Layers::GlobalAveragePooling1D>(FP32)
         .add<Layers::Dense>(CHANNELS, 32, FP32)
         .add<Layers::ReLU>()
         .add<Layers::Dense>(32, NUM_CLASSES, FP32)
         .add<Layers::Softmax>(FP32);
    model.summary();
    printf("Model size: %zu bytes\n\n", model.get_bytes());

    // Shuffle training data
    {
        std::mt19937 rng(42);
        std::vector<size_t> idx(N);
        std::iota(idx.begin(), idx.end(), 0);
        std::ranges::shuffle(idx, rng);
        std::vector<float> Xs(X.size());
        std::vector<int>   ys(N);
        for (size_t i = 0; i < idx.size(); i++) {
            size_t src = idx[i];
            std::copy(X.begin() + src * TIMESTEPS * CHANNELS,
                      X.begin() + (src + 1) * TIMESTEPS * CHANNELS,
                      Xs.begin() + i * TIMESTEPS * CHANNELS);
            ys[i] = y[src];
        }
        X = std::move(Xs);
        y = std::move(ys);
    }

    constexpr float lr     = 0.01f;
    constexpr int   epochs = 10;
    printf("Training  lr=%.3f  epochs=%d  samples=%d\n\n", lr, epochs, N);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        int correct = 0;

        for (int s = 0; s < static_cast<int>(y.size()); s++) {
            Tensors::Tensor input({static_cast<size_t>(CHANNELS), static_cast<size_t>(TIMESTEPS)}, FP32);
            for (int t = 0; t < TIMESTEPS; t++)
                for (int f = 0; f < CHANNELS; f++)
                    input.set({static_cast<size_t>(f), static_cast<size_t>(t)},
                              X[s * TIMESTEPS * CHANNELS + t * CHANNELS + f]);

            Tensors::Tensor target({static_cast<size_t>(NUM_CLASSES)}, FP32);
            target.all_zeros();
            target.set({static_cast<size_t>(y[s])}, 1.0);

            auto output = model.forward(input);
            total_loss += Loss::CrossEntropy::forward(output, target);

            int pred = 0;
            for (int c = 1; c < NUM_CLASSES; c++)
                if (output.at({static_cast<size_t>(c)}) > output.at({static_cast<size_t>(pred)})) pred = c;
            if (pred == y[s]) correct++;

            auto grad = Loss::CrossEntropy::backward(output, target, FP32);
            model.backward(grad, lr);
        }

        printf("Epoch %2d  loss: %.4f  train_acc: %.1f%%\n",
               epoch + 1, total_loss / N, 100.0 * correct / static_cast<double>(N));
    }

    // Evaluate
    const char* CLASS_NAMES[4] = {"idle", "walking", "fidget", "drop"};
    printf("\nEvaluating on %d test samples...\n", test.N);
    int total_correct = 0;
    int class_correct[4] = {}, class_total[4] = {};

    for (int s = 0; s < static_cast<int>(test.y.size()); s++) {
        Tensors::Tensor input({static_cast<size_t>(CHANNELS), static_cast<size_t>(TIMESTEPS)}, FP32);
        for (int t = 0; t < TIMESTEPS; t++)
            for (int f = 0; f < CHANNELS; f++)
                input.set({static_cast<size_t>(f), static_cast<size_t>(t)},
                          test.X[s * TIMESTEPS * CHANNELS + t * CHANNELS + f]);

        auto output = model.forward(input);

        int pred = 0;
        for (int c = 1; c < NUM_CLASSES; c++)
            if (output.at({static_cast<size_t>(c)}) > output.at({static_cast<size_t>(pred)})) pred = c;

        class_total[test.y[s]]++;
        if (pred == test.y[s]) { total_correct++; class_correct[test.y[s]]++; }
    }

    printf("Test accuracy: %d/%d = %.1f%%\n\n",
           total_correct, test.N, 100.0 * total_correct / static_cast<double>(test.N));
    printf("%-12s  Correct / Total\n", "Class");
    printf("------------------------------------------\n");
    for (int c = 0; c < NUM_CLASSES; c++) {
        if (class_total[c] > 0)
            printf("  %-10s  %3d / %-3d  (%.0f%%)\n",
                   CLASS_NAMES[c], class_correct[c], class_total[c],
                   100.0 * class_correct[c] / static_cast<double>(class_total[c]));
    }
    printf("Model Size: %ld Bytes (%ld KB)", model.get_bytes(), model.get_bytes() / 1024);
    return 0;
}