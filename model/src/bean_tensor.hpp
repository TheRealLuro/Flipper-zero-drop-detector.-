#pragma once
// Auto-extracted from main.cpp lines 1-1022 — BeanTensor library + dataset/JSON/CSV/predict helpers.
// Kept identical so main.cpp continues to build standalone.

#define BT_MAX_LAYERS 32
#define BT_L_MAX_DIMS 3
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
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
        BT_INT32,
        BT_INT64,
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
            case BT_INT32: return sizeof(int32_t);
            case BT_INT64: return sizeof(int64_t);
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
                    case BT_INT32: ptr[offset] = static_cast<int32_t>(value);   break;
                    case BT_INT64: ptr[offset] = static_cast<int64_t>(value);   break;
                }
            }, this->_data);
        }

        void set_random(const double min = 0, const double max = 1, const size_t seed = 42) {
            if (min >= max) throw std::invalid_argument("Min must be less than max");
            switch (this->_precision) {
                case FP32:  if (min > std::numeric_limits<float>::max()   || max < std::numeric_limits<float>::lowest())   throw std::invalid_argument("Min or max out of range for FP32");  break;
                case FP64:  if (min > std::numeric_limits<double>::max()  || max < std::numeric_limits<double>::lowest())  throw std::invalid_argument("Min or max out of range for FP64");  break;
                case BT_INT32: if (min > std::numeric_limits<int32_t>::max() || max < std::numeric_limits<int32_t>::min())    throw std::invalid_argument("Min or max out of range for BT_INT32"); break;
                case BT_INT64: if (min > static_cast<double>(std::numeric_limits<int64_t>::max()) || max < static_cast<double>(std::numeric_limits<int64_t>::min())) throw std::invalid_argument("Min or max out of range for BT_INT64"); break;
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
                        case BT_INT32: ptr[i] = static_cast<int32_t>(0);   break;
                        case BT_INT64: ptr[i] = static_cast<int64_t>(0);   break;
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
                        case BT_INT32: ptr[i] = static_cast<int32_t>(1);   break;
                        case BT_INT64: ptr[i] = static_cast<int64_t>(1);   break;
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
            if (precision != FP32 && precision != FP64 && precision != BT_INT32 && precision != BT_INT64)
                throw std::invalid_argument("Invalid precision");
            this->_dims_l = dims.size();
            for (size_t i = 0; i < this->_dims_l; i++) this->_dims[i] = dims[i];
            this->_precision = precision;
            switch (this->_precision) {
                case FP32:  this->_data = new float[this->numel()];    break;
                case FP64:  this->_data = new double[this->numel()];   break;
                case BT_INT32: this->_data = new int32_t[this->numel()];  break;
                case BT_INT64: this->_data = new int64_t[this->numel()];  break;
            }
        }

        Tensor(const std::initializer_list<size_t>& dims, const Precision& precision) {
            if (dims.size() > BT_L_MAX_DIMS) throw std::invalid_argument("Tensor dimension exceeds maximum supported dimensions");
            if (dims.size() == 0)            throw std::invalid_argument("Tensor must have at least one dimension");
            if (precision != FP32 && precision != FP64 && precision != BT_INT32 && precision != BT_INT64)
                throw std::invalid_argument("Invalid precision");
            this->_dims_l = dims.size();
            for (size_t i = 0; i < this->_dims_l; i++) this->_dims[i] = dims.begin()[i];
            this->_precision = precision;
            switch (this->_precision) {
                case FP32:  this->_data = new float[this->numel()];    break;
                case FP64:  this->_data = new double[this->numel()];   break;
                case BT_INT32: this->_data = new int32_t[this->numel()];  break;
                case BT_INT64: this->_data = new int64_t[this->numel()];  break;
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
        virtual void dump_params(std::ostringstream& /*os*/, const std::string& /*key*/) const {}
        [[nodiscard]] virtual bool has_params() const { return false; }
        std::optional<Tensors::Tensor> _last_input;
    };

    // Accepts any rank; ReLU is shape-agnostic.
    struct ReLU : Layer {
        Tensors::Tensor forward(const Tensors::Tensor& input) override {
            _last_input = input;
            auto output = Tensors::Tensor(input);
            for (size_t i = 0; i < output.numel(); i++) {
                double v = std::visit([i](const auto* ptr) -> double { return static_cast<double>(ptr[i]); }, output._data);
                if (v < 0.0) {
                    std::visit([i](auto* ptr) { ptr[i] = 0; }, output._data);
                }
            }
            return output;
        }

        Tensors::Tensor backward(const Tensors::Tensor& grad_output, float32_t lr) override {
            if (!_last_input) throw std::runtime_error("forward() must be called before backward()");
            std::vector<size_t> dims(_last_input->_dims, _last_input->_dims + _last_input->_dims_l);
            auto d_input = Tensors::Tensor(dims, _last_input->_precision);
            const auto& in = *_last_input;
            for (size_t i = 0; i < d_input.numel(); i++) {
                double x  = std::visit([i](const auto* p) -> double { return static_cast<double>(p[i]); }, in._data);
                double go = std::visit([i](const auto* p) -> double { return static_cast<double>(p[i]); }, grad_output._data);
                std::visit([i, x, go](auto* p) {
                    using T = std::remove_pointer_t<decltype(p)>;
                    p[i] = static_cast<T>(x < 0.0 ? 0.0 : go);
                }, d_input._data);
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

        explicit Dense(const uint32_t in_features, const uint32_t out_features, const Precision& precision, size_t seed = std::random_device{}()) :
            _weight({out_features, in_features}, precision),
            _bias({out_features}, precision) {
            this->_in_features  = in_features;
            this->_out_features = out_features;
            this->_precision    = precision;
            // Xavier uniform: keeps activation variance stable across layers
            const double limit = std::sqrt(6.0 / (in_features + out_features));
            this->_weight.set_random(-limit, limit, seed);
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
            return detail::get_precision_length(_precision) * (_weight.numel() + _bias.numel());
        }
        [[nodiscard]] bool has_params() const override { return true; }
        void dump_params(std::ostringstream& os, const std::string& key) const override {
            os << "  \"" << key << "\": {\n";
            os << "    \"weight_shape\": [" << _out_features << ", " << _in_features << "],\n";
            os << "    \"weight\": [";
            for (size_t i = 0; i < _weight.numel(); i++) {
                if (i) os << ", ";
                double v = std::visit([i](const auto* p) -> double { return static_cast<double>(p[i]); }, _weight._data);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.9g", v);
                os << buf;
            }
            os << "],\n";
            os << "    \"bias_shape\": [" << _out_features << "],\n";
            os << "    \"bias\": [";
            for (size_t i = 0; i < _bias.numel(); i++) {
                if (i) os << ", ";
                double v = std::visit([i](const auto* p) -> double { return static_cast<double>(p[i]); }, _bias._data);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.9g", v);
                os << buf;
            }
            os << "]\n";
            os << "  }";
        }
    };

    // Conv1D — valid convolution over a {in_channels, T} tensor.
    // Weight layout: {out_channels, in_channels, kernel}. Output: {out_channels, L_out}.
    // L_out = (T - kernel) / stride + 1
    struct Conv1D : Layer {
        uint32_t _in_channels;
        uint32_t _out_channels;
        uint32_t _kernel;
        uint32_t _stride;
        Tensors::Tensor _weight;
        Tensors::Tensor _bias;
        Precision _precision;

        explicit Conv1D(uint32_t in_channels, uint32_t out_channels, uint32_t kernel, uint32_t stride,
                        const Precision& precision, size_t seed = std::random_device{}()) :
            _in_channels(in_channels), _out_channels(out_channels),
            _kernel(kernel), _stride(stride),
            _weight({out_channels, in_channels, kernel}, precision),
            _bias({out_channels}, precision),
            _precision(precision) {
            if (kernel == 0 || stride == 0) throw std::invalid_argument("Conv1D kernel/stride must be > 0");
            // He uniform: limit = sqrt(6 / fan_in), fan_in = in_channels * kernel.
            const double fan_in = static_cast<double>(in_channels) * kernel;
            const double limit  = std::sqrt(6.0 / fan_in);
            _weight.set_random(-limit, limit, seed);
            _bias.all_zeros();
        }

        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& input) override {
            if (input._dims_l != 2) throw std::invalid_argument("Conv1D expects {in_channels, T} input");
            if (input._dims[0] != _in_channels) throw std::invalid_argument("Conv1D in_channels mismatch");
            const uint32_t T = input._dims[1];
            if (T < _kernel) throw std::invalid_argument("Conv1D input shorter than kernel");
            const uint32_t L_out = (T - _kernel) / _stride + 1;
            _last_input = input;
            auto output = Tensors::Tensor({_out_channels, L_out}, _precision);
            output.all_zeros();
            for (uint32_t oc = 0; oc < _out_channels; oc++) {
                const double b = _bias.at({oc});
                for (uint32_t p = 0; p < L_out; p++) {
                    double acc = 0.0;
                    const uint32_t t0 = p * _stride;
                    for (uint32_t ic = 0; ic < _in_channels; ic++) {
                        for (uint32_t k = 0; k < _kernel; k++) {
                            acc += _weight.at({oc, ic, k}) * input.at({ic, t0 + k});
                        }
                    }
                    output.set({oc, p}, acc + b);
                }
            }
            return output;
        }

        [[nodiscard]] Tensors::Tensor backward(const Tensors::Tensor& grad_output, float32_t lr) override {
            if (!_last_input) throw std::runtime_error("forward() must be called before backward()");
            const auto& input = *_last_input;
            const uint32_t T = input._dims[1];
            const uint32_t L_out = grad_output._dims[1];

            std::vector<double> dW(_weight.numel(), 0.0);
            std::vector<double> db(_out_channels, 0.0);
            auto d_input = Tensors::Tensor({_in_channels, T}, _precision);
            d_input.all_zeros();
            std::vector<double> d_input_acc(static_cast<size_t>(_in_channels) * T, 0.0);

            auto w_idx = [&](uint32_t oc, uint32_t ic, uint32_t k) {
                return ((static_cast<size_t>(oc) * _in_channels) + ic) * _kernel + k;
            };
            auto x_idx = [&](uint32_t ic, uint32_t t) {
                return static_cast<size_t>(ic) * T + t;
            };

            for (uint32_t oc = 0; oc < _out_channels; oc++) {
                for (uint32_t p = 0; p < L_out; p++) {
                    const double go = grad_output.at({oc, p});
                    db[oc] += go;
                    const uint32_t t0 = p * _stride;
                    for (uint32_t ic = 0; ic < _in_channels; ic++) {
                        for (uint32_t k = 0; k < _kernel; k++) {
                            const double w = _weight.at({oc, ic, k});
                            const double x = input.at({ic, t0 + k});
                            dW[w_idx(oc, ic, k)] += go * x;
                            d_input_acc[x_idx(ic, t0 + k)] += go * w;
                        }
                    }
                }
            }

            for (uint32_t ic = 0; ic < _in_channels; ic++)
                for (uint32_t t = 0; t < T; t++)
                    d_input.set({ic, t}, d_input_acc[x_idx(ic, t)]);

            for (uint32_t oc = 0; oc < _out_channels; oc++) {
                for (uint32_t ic = 0; ic < _in_channels; ic++) {
                    for (uint32_t k = 0; k < _kernel; k++) {
                        const double cur = _weight.at({oc, ic, k});
                        _weight.set({oc, ic, k}, cur - lr * dW[w_idx(oc, ic, k)]);
                    }
                }
                _bias.set({oc}, _bias.at({oc}) - lr * db[oc]);
            }

            return d_input;
        }

        [[nodiscard]] std::string name() const override { return "Conv1D"; }
        [[nodiscard]] size_t param_bytes() const override {
            return detail::get_precision_length(_precision) * (_weight.numel() + _bias.numel());
        }
        [[nodiscard]] bool has_params() const override { return true; }
        void dump_params(std::ostringstream& os, const std::string& key) const override {
            os << "  \"" << key << "\": {\n";
            os << "    \"weight_shape\": [" << _out_channels << ", " << _in_channels << ", " << _kernel << "],\n";
            os << "    \"weight\": [";
            for (size_t i = 0; i < _weight.numel(); i++) {
                if (i) os << ", ";
                double v = std::visit([i](const auto* p) -> double { return static_cast<double>(p[i]); }, _weight._data);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.9g", v);
                os << buf;
            }
            os << "],\n";
            os << "    \"bias_shape\": [" << _out_channels << "],\n";
            os << "    \"bias\": [";
            for (size_t i = 0; i < _bias.numel(); i++) {
                if (i) os << ", ";
                double v = std::visit([i](const auto* p) -> double { return static_cast<double>(p[i]); }, _bias._data);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.9g", v);
                os << buf;
            }
            os << "]\n";
            os << "  }";
        }
    };

    // Global max-pool over the last (time) axis of a {channels, T} tensor.
    struct GlobalMaxPool1D : Layer {
        Precision _precision;
        std::vector<size_t> _argmax;  // one argmax per channel, set during forward

        explicit GlobalMaxPool1D(const Precision& precision) : _precision(precision) {}

        [[nodiscard]] Tensors::Tensor forward(const Tensors::Tensor& input) override {
            if (input._dims_l != 2) throw std::invalid_argument("GlobalMaxPool1D expects {channels, T} input");
            const uint32_t channels = input._dims[0];
            const uint32_t length   = input._dims[1];
            _last_input = input;
            _argmax.assign(channels, 0);
            auto output = Tensors::Tensor({channels}, _precision);
            for (uint32_t ch = 0; ch < channels; ch++) {
                double best = input.at({ch, 0});
                size_t best_pos = 0;
                for (uint32_t t = 1; t < length; t++) {
                    double v = input.at({ch, t});
                    if (v > best) { best = v; best_pos = t; }
                }
                _argmax[ch] = best_pos;
                output.set({ch}, best);
            }
            return output;
        }

        [[nodiscard]] Tensors::Tensor backward(const Tensors::Tensor& grad_output, float32_t /*lr*/) override {
            if (!_last_input) throw std::runtime_error("forward() must be called before backward()");
            const uint32_t channels = _last_input->_dims[0];
            const uint32_t length   = _last_input->_dims[1];
            auto d_input = Tensors::Tensor({channels, length}, _precision);
            d_input.all_zeros();
            for (uint32_t ch = 0; ch < channels; ch++) {
                d_input.set({ch, _argmax[ch]}, grad_output.at({ch}));
            }
            return d_input;
        }

        [[nodiscard]] std::string name() const override { return "GlobalMaxPool1D"; }
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
        std::vector<float> mean;  // shape [F], from train
        std::vector<float> std;   // shape [F], from train
        std::vector<float> X;     // shape [N][T][F], row-major, already normalized
        std::vector<int>   y;     // shape [N]
    };

    static void must_read(void* dst, size_t elem_size, size_t n, FILE* f, const char* path, const char* what) {
        size_t got = fread(dst, elem_size, n, f);
        if (got != n) {
            char msg[512];
            std::snprintf(msg, sizeof(msg), "Short read on %s (%s): wanted %zu, got %zu", path, what, n, got);
            throw std::runtime_error(msg);
        }
    }

    Dataset load_bin(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) {
            char msg[512];
            std::snprintf(msg, sizeof(msg), "Could not open dataset file: %s", path);
            throw std::runtime_error(msg);
        }
        Dataset d;
        must_read(&d.N, sizeof(int32_t), 1, f, path, "N");
        must_read(&d.T, sizeof(int32_t), 1, f, path, "T");
        must_read(&d.F, sizeof(int32_t), 1, f, path, "F");
        if (d.N < 0 || d.T < 0 || d.F < 0) { fclose(f); throw std::runtime_error("Negative dimension in dataset header"); }
        d.mean.resize(d.F);
        d.std.resize(d.F);
        must_read(d.mean.data(), sizeof(float), d.mean.size(), f, path, "mean");
        must_read(d.std.data(),  sizeof(float), d.std.size(),  f, path, "std");
        d.X.resize(static_cast<size_t>(d.N) * d.T * d.F);
        d.y.resize(d.N);
        must_read(d.X.data(), sizeof(float),   d.X.size(), f, path, "X");
        must_read(d.y.data(), sizeof(int32_t), d.y.size(), f, path, "y");
        fclose(f);
        return d;
    }

}
using namespace BeanTensor::lite;

// Fills `dst` ({channels, timesteps}) from a flat [T][F] sample in `src` at `s * T * F`.
static void load_sample(Tensors::Tensor& dst, const std::vector<float>& src,
                        size_t s, int T, int F) {
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            dst.set({static_cast<size_t>(f), static_cast<size_t>(t)},
                    src[s * T * F + t * F + f]);
}

static void print_normalization_stats(const std::vector<float>& mean,
                                      const std::vector<float>& std) {
    printf("\n// Per-feature normalization (a_mag_sq, g_mag_sq, a_mag_sq_delta, g_mag_sq_delta)\n");
    printf("static const float FEATURE_MEAN[%zu]    = { ", mean.size());
    for (size_t i = 0; i < mean.size(); i++) {
        if (i) printf(", ");
        printf("%.9gf", mean[i]);
    }
    printf(" };\n");
    printf("static const float FEATURE_INV_STD[%zu] = { ", std.size());
    for (size_t i = 0; i < std.size(); i++) {
        if (i) printf(", ");
        printf("%.9gf", 1.0f / std[i]);
    }
    printf(" };\n\n");
}

// Architecture spec: full parameterization of the Conv1D + Dense head topology.
// Defaults match the original baseline so an old call site keeps the same model.
struct ArchSpec {
    uint32_t conv_channels = 8;
    uint32_t conv_kernel   = 7;
    uint32_t conv_stride   = 2;
    uint32_t hidden_layers = 0;   // # Dense+ReLU blocks between pool and output (0..3)
    uint32_t hidden_units  = 16;  // units per hidden Dense layer
};

// Builds: Conv1D -> ReLU -> GlobalMaxPool1D -> (Dense -> ReLU)* -> Dense -> Softmax.
// Same shape must be used for training, predict, and tune so weights match.
static Layers::Sequential build_model(uint32_t in_channels, uint32_t num_classes,
                                      const ArchSpec& a = ArchSpec{}) {
    Layers::Sequential model(FP32);
    model.add<Layers::Conv1D>(in_channels, a.conv_channels, a.conv_kernel, a.conv_stride, FP32)
         .add<Layers::ReLU>()
         .add<Layers::GlobalMaxPool1D>(FP32);
    uint32_t prev = a.conv_channels;
    for (uint32_t i = 0; i < a.hidden_layers; i++) {
        model.add<Layers::Dense>(prev, a.hidden_units, FP32)
             .add<Layers::ReLU>();
        prev = a.hidden_units;
    }
    model.add<Layers::Dense>(prev, num_classes, FP32)
         .add<Layers::Softmax>(FP32);
    return model;
}

static std::string read_text_file(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("Could not open " + p.string());
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

// Reads the float array at `obj_key.array_key` from a JSON string written by
// export_model_json. Format is well-known so we don't need a full parser.
static std::vector<float> json_floats(const std::string& json,
                                      const std::string& obj_key,
                                      const std::string& array_key) {
    const std::string obj_marker = "\"" + obj_key + "\"";
    size_t o = json.find(obj_marker);
    if (o == std::string::npos) throw std::runtime_error("JSON missing object: " + obj_key);
    const std::string arr_marker = "\"" + array_key + "\"";
    size_t k = json.find(arr_marker, o);
    if (k == std::string::npos) throw std::runtime_error("JSON missing key: " + obj_key + "." + array_key);
    size_t lb = json.find('[', k);
    size_t rb = json.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos)
        throw std::runtime_error("JSON malformed array for " + obj_key + "." + array_key);
    std::vector<float> out;
    size_t p = lb + 1;
    while (p < rb) {
        while (p < rb && (json[p] == ' ' || json[p] == ',' || json[p] == '\n' || json[p] == '\t' || json[p] == '\r')) p++;
        if (p >= rb) break;
        char* end = nullptr;
        float v = std::strtof(json.data() + p, &end);
        if (end == json.data() + p) throw std::runtime_error("JSON: failed to parse float in " + obj_key + "." + array_key);
        out.push_back(v);
        p = end - json.data();
    }
    return out;
}

// Copies a flat float buffer into the given tensor's storage. Caller must ensure sizes match.
static void tensor_assign_flat(Tensors::Tensor& t, const std::vector<float>& src) {
    if (src.size() != t.numel())
        throw std::runtime_error("tensor_assign_flat: size mismatch");
    std::visit([&](auto* ptr) {
        using T = std::remove_pointer_t<decltype(ptr)>;
        for (size_t i = 0; i < src.size(); i++) ptr[i] = static_cast<T>(src[i]);
    }, t._data);
}

// Reads a single integer field from `"<obj_key>": { ... "<field>": N ... }` in JSON.
// Returns `dflt` if the object or field isn't present.
static int json_int_field(const std::string& json,
                          const std::string& obj_key,
                          const std::string& field,
                          int dflt) {
    const std::string obj_marker = "\"" + obj_key + "\"";
    size_t o = json.find(obj_marker);
    if (o == std::string::npos) return dflt;
    const std::string field_marker = "\"" + field + "\"";
    size_t k = json.find(field_marker, o);
    if (k == std::string::npos) return dflt;
    size_t colon = json.find(':', k);
    if (colon == std::string::npos) return dflt;
    size_t p = colon + 1;
    while (p < json.size() && std::isspace(static_cast<unsigned char>(json[p]))) p++;
    if (p >= json.size()) return dflt;
    char* end = nullptr;
    long v = std::strtol(json.data() + p, &end, 10);
    if (end == json.data() + p) return dflt;
    return static_cast<int>(v);
}

// Reads the ArchSpec out of a model.json. Falls back to defaults when the file
// was written by an older version that didn't include an "arch" block.
static ArchSpec read_arch_from_json(const std::filesystem::path& json_path) {
    ArchSpec a;
    std::string json = read_text_file(json_path);
    a.conv_channels = (uint32_t)json_int_field(json, "arch", "conv_channels", (int)a.conv_channels);
    a.conv_kernel   = (uint32_t)json_int_field(json, "arch", "conv_kernel",   (int)a.conv_kernel);
    a.conv_stride   = (uint32_t)json_int_field(json, "arch", "conv_stride",   (int)a.conv_stride);
    a.hidden_layers = (uint32_t)json_int_field(json, "arch", "hidden_layers", (int)a.hidden_layers);
    a.hidden_units  = (uint32_t)json_int_field(json, "arch", "hidden_units",  (int)a.hidden_units);
    return a;
}

// Loads weights from model.json into the given model in layer order.
// Conv1D weights come from "conv1d"; Dense layers are read in order as "dense_0",
// "dense_1", ... The legacy single-Dense format ("dense") is also accepted.
static void load_weights_from_json(Layers::Sequential& model, const std::filesystem::path& json_path) {
    const std::string json = read_text_file(json_path);
    int dense_idx = 0;
    bool legacy_dense_used = false;
    for (auto& layer : model._layers) {
        if (auto* c = dynamic_cast<Layers::Conv1D*>(layer.get())) {
            auto w = json_floats(json, "conv1d", "weight");
            auto b = json_floats(json, "conv1d", "bias");
            tensor_assign_flat(c->_weight, w);
            tensor_assign_flat(c->_bias,   b);
        } else if (auto* d = dynamic_cast<Layers::Dense*>(layer.get())) {
            std::string key = "dense_" + std::to_string(dense_idx);
            // Try indexed key first; fall back to legacy "dense" if absent and not already used.
            std::string probe = "\"" + key + "\"";
            if (json.find(probe) != std::string::npos) {
                auto w = json_floats(json, key, "weight");
                auto b = json_floats(json, key, "bias");
                tensor_assign_flat(d->_weight, w);
                tensor_assign_flat(d->_bias,   b);
            } else if (!legacy_dense_used) {
                auto w = json_floats(json, "dense", "weight");
                auto b = json_floats(json, "dense", "bias");
                tensor_assign_flat(d->_weight, w);
                tensor_assign_flat(d->_bias,   b);
                legacy_dense_used = true;
            } else {
                throw std::runtime_error("model.json missing weights for " + key);
            }
            dense_idx++;
        }
    }
}

// ---------- CSV / feature extraction for predict mode ----------

// Reads a drop_detect/*.csv. Each row: order, ax, ay, az, gx, gy, gz (header optional).
// Returns rows of 6 floats: [ax, ay, az, gx, gy, gz].
static std::vector<std::array<float, 6>> read_imu_csv(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Could not open CSV: " + path.string());
    std::vector<std::array<float, 6>> rows;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            // Skip header if first cell isn't a number.
            if (!line.empty() && (line[0] < '0' || line[0] > '9') && line[0] != '-' && line[0] != '+') continue;
        }
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> parts;
        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) { parts.push_back(0.0f); continue; }
            try { parts.push_back(std::stof(cell)); }
            catch (...) { parts.clear(); break; }
        }
        if (parts.size() < 7) continue;  // need order + 6 channels
        rows.push_back({parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]});
    }
    return rows;
}

// Build a (C=4, T=249) tensor for window starting at `start` in raw IMU rows.
// Features per timestep t (1..WINDOW-1, where WINDOW=250):
//   f0: ax^2 + ay^2 + az^2
//   f1: gx^2 + gy^2 + gz^2
//   f2: f0(t) - f0(t-1)
//   f3: f1(t) - f1(t-1)
// Then per-feature normalization (x - mean) * inv_std using the same stats data.py produced.
static Tensors::Tensor make_window_tensor(const std::vector<std::array<float, 6>>& rows,
                                          size_t start,
                                          const std::vector<float>& mean,
                                          const std::vector<float>& std_) {
    constexpr int WINDOW = 250;
    constexpr int T = WINDOW - 1;  // 249 feature vectors
    constexpr int F = 4;
    auto sq = [](float a, float b, float c) { return a*a + b*b + c*c; };

    Tensors::Tensor out({static_cast<size_t>(F), static_cast<size_t>(T)}, FP32);
    out.all_zeros();
    for (int t = 1; t < WINDOW; t++) {
        const auto& cur  = rows[start + t];
        const auto& prev = rows[start + t - 1];
        float a      = sq(cur[0],  cur[1],  cur[2]);
        float g      = sq(cur[3],  cur[4],  cur[5]);
        float a_prev = sq(prev[0], prev[1], prev[2]);
        float g_prev = sq(prev[3], prev[4], prev[5]);
        float feats[4] = { a, g, a - a_prev, g - g_prev };
        for (int f = 0; f < F; f++) {
            float normed = (feats[f] - mean[f]) / std_[f];
            out.set({static_cast<size_t>(f), static_cast<size_t>(t - 1)}, normed);
        }
    }
    return out;
}

// Run prediction on every WINDOW-sized window of a CSV, stride=125 (matches data.py).
// Loads weights from model.json and normalization stats from TRAIN.bin in the dataset_dir.
static int run_predict(const std::filesystem::path& csv_path,
                       const std::filesystem::path& model_json,
                       const std::filesystem::path& norm_bin) {
    constexpr int WINDOW = 250;
    constexpr int STRIDE = 125;
    constexpr uint32_t CHANNELS = 4;
    constexpr uint32_t NUM_CLASSES = 4;
    const char* CLASS_NAMES[4] = {"idle", "walking", "fidget", "drop"};

    auto rows = read_imu_csv(csv_path);
    if (rows.size() < WINDOW) {
        fprintf(stderr, "CSV has only %zu rows; need at least %d for one window.\n", rows.size(), WINDOW);
        return 1;
    }
    printf("CSV: %s  (%zu rows)\n", csv_path.string().c_str(), rows.size());

    auto norm = DatasetOpener::load_bin(norm_bin.string().c_str());
    printf("Normalization stats from %s  (mean[0]=%.4g, std[0]=%.4g)\n",
           norm_bin.string().c_str(), norm.mean[0], norm.std[0]);

    ArchSpec arch = read_arch_from_json(model_json);
    auto model = build_model(CHANNELS, NUM_CLASSES, arch);
    load_weights_from_json(model, model_json);
    printf("Loaded weights from %s\n", model_json.string().c_str());
    model.summary();
    printf("\n");

    printf("%-8s %-14s  %-7s %-7s %-7s %-7s   -> verdict\n",
           "Window", "Rows", "idle", "walking", "fidget", "drop");
    printf("--------------------------------------------------------------------\n");

    int counts[4] = {};
    int n_windows = 0;
    for (size_t start = 0; start + WINDOW <= rows.size(); start += STRIDE) {
        auto input = make_window_tensor(rows, start, norm.mean, norm.std);
        auto probs = model.forward(input);

        int pred = 0;
        for (uint32_t c = 1; c < NUM_CLASSES; c++)
            if (probs.at({c}) > probs.at({static_cast<size_t>(pred)})) pred = static_cast<int>(c);

        printf("%-8d %5zu-%-7zu  ", n_windows, start, start + WINDOW - 1);
        for (uint32_t c = 0; c < NUM_CLASSES; c++) printf("%-7.3f ", probs.at({c}));
        printf("  -> %s%s\n", CLASS_NAMES[pred], (pred == 3) ? "  <-- DROP" : "");

        counts[pred]++;
        n_windows++;
    }

    printf("\nSummary across %d windows:\n", n_windows);
    for (int c = 0; c < NUM_CLASSES; c++)
        printf("  %-8s %3d windows (%.0f%%)\n", CLASS_NAMES[c], counts[c],
               n_windows ? 100.0 * counts[c] / n_windows : 0.0);
    return 0;
}

static void export_model_json(const Layers::Sequential& model,
                              const std::filesystem::path& out_path,
                              const ArchSpec* arch = nullptr) {
    std::ostringstream os;
    os << "{\n";

    // Architecture block first so predict/tune can rebuild the matching shape.
    if (arch) {
        os << "  \"arch\": {\n";
        os << "    \"conv_channels\": " << arch->conv_channels << ",\n";
        os << "    \"conv_kernel\": "   << arch->conv_kernel   << ",\n";
        os << "    \"conv_stride\": "   << arch->conv_stride   << ",\n";
        os << "    \"hidden_layers\": " << arch->hidden_layers << ",\n";
        os << "    \"hidden_units\": "  << arch->hidden_units  << "\n";
        os << "  },\n";
    }

    bool first = true;
    int dense_idx = 0;
    for (const auto& layer : model._layers) {
        if (!layer->has_params()) continue;
        if (!first) os << ",\n";
        first = false;
        // Conv1D keeps the unique "conv1d" key; Dense layers get enumerated.
        std::string key;
        if (layer->name() == "Conv1D") key = "conv1d";
        else if (layer->name() == "Dense") { key = "dense_" + std::to_string(dense_idx++); }
        else {
            key = layer->name();
            for (auto& c : key) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        layer->dump_params(os, key);
    }
    os << "\n}\n";

    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("Could not open " + out_path.string() + " for write");
    f << os.str();
    f.close();
    printf("Wrote weights to %s\n", out_path.string().c_str());
}
