#ifndef LINALG_H
#define LINALG_H

#include "tensor.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace linalg {

    template<typename T>
    struct MinMax {
        T min;
        T max;
    };

    template<typename T, size_t N>
    inline bool shapesMatch(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
        return std::memcmp(t1.getShape(), t2.getShape(), N * sizeof(size_t)) == 0;
    }

    template<typename T, size_t N>
    Tensor<T, N> add(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
        Tensor<T, N> t(reinterpret_cast<const size_t(&)[N]>(*t1.getShape()));
        if (!shapesMatch(t1, t2)) {
            std::printf("Shapes do not match\n");
            return t;
        }
        for (size_t i = 0; i < t.getLength(); i++) {
            t[i] = t1[i] + t2[i];
        }
        return t;
    }

    template<typename T, size_t N>
    Tensor<T, N> subtract(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
        Tensor<T, N> t(reinterpret_cast<const size_t(&)[N]>(*t1.getShape()));
        if (!shapesMatch(t1, t2)) {
            std::printf("Shapes do not match\n");
            return t;
        }
        for (size_t i = 0; i < t.getLength(); i++) {
            t[i] = t1[i] - t2[i];
        }
        return t;
    }

    template<typename T, size_t N>
    MinMax<T> minAndMax(const Tensor<T, N>& t) {
        if (t.getLength() == 0) return { T{}, T{} };
        T min = t[0];
        T max = t[0];
        for (size_t i = 1; i < t.getLength(); i++) {
            if (t[i] < min) min = t[i];
            if (t[i] > max) max = t[i];
        }
        return MinMax<T>{ min, max };
    }

    template<typename T, size_t N>
    float mean(const Tensor<T, N>& t) {
        if (t.getLength() == 0) return 0.0f;
        float mu = 0.0f;
        for (size_t i = 0; i < t.getLength(); i++) {
            mu += static_cast<float>(t[i]);
        }
        return mu / static_cast<float>(t.getLength());
    }

    template<typename T, size_t N>
    float variance(const Tensor<T, N>& t, float mu = 0.0f, bool compute_mean = true) {
        if (t.getLength() == 0) return 0.0f;
        if (compute_mean) {
            mu = mean(t);
        }
        float var = 0.0f;
        for (size_t i = 0; i < t.getLength(); i++) {
            float diff = static_cast<float>(t[i]) - mu;
            var += diff * diff;
        }
        return var / static_cast<float>(t.getLength());
    }

    template<typename T, size_t N>
    float standardDeviation(const Tensor<T, N>& t) {
        return std::sqrt(variance(t));
    }

    template<typename T, size_t N>
    Tensor<T, N> normalizeMinMax(const Tensor<T, N>& tensor) {
        MinMax<T> minmax = minAndMax(tensor);
        Tensor<T, N> norm(reinterpret_cast<const size_t(&)[N]>(*tensor.getShape()));
        T range = minmax.max - minmax.min;
        if (range == T{}) return norm;

        for (size_t i = 0; i < tensor.getLength(); i++) {
            norm[i] = (tensor[i] - minmax.min) / range;
        }
        return norm;
    }

    template<typename T, size_t N>
    Tensor<T, N> normalizeZScore(const Tensor<T, N>& t) {
        Tensor<T, N> norm(reinterpret_cast<const size_t(&)[N]>(*t.getShape()));
        float mu = mean(t);
        float std_d = standardDeviation(t);
        if (std_d < std::numeric_limits<float>::epsilon()) return norm;

        for (size_t i = 0; i < t.getLength(); i++) {
            norm[i] = static_cast<T>((static_cast<float>(t[i]) - mu) / std_d);
        }
        return norm;
    }

    template<typename T, size_t N>
    float dot(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
        if (!shapesMatch(t1, t2)) {
            std::printf("Shapes do not match.\n");
            return 0.0f;
        }
        float dotted = 0.0f;
        for (size_t i = 0; i < t1.getLength(); i++) {
            dotted += static_cast<float>(t1[i]) * static_cast<float>(t2[i]);
        }
        return dotted;
    }

    template<typename T, size_t N>
    float cosineSimilarity(const Tensor<T, N>& t1, const Tensor<T, N>& t2) {
        if (!shapesMatch(t1, t2)) {
            std::printf("Shapes do not match.\n");
            return 0.0f;
        }
        float dot_prod = 0.0f;
        float n1 = 0.0f;
        float n2 = 0.0f;
        for (size_t i = 0; i < t1.getLength(); i++) {
            float t1_val = static_cast<float>(t1[i]);
            float t2_val = static_cast<float>(t2[i]);
            dot_prod += t1_val * t2_val;
            n1 += t1_val * t1_val;
            n2 += t2_val * t2_val;
        }
        float denominator = std::sqrt(n1) * std::sqrt(n2);
        if (denominator < std::numeric_limits<float>::epsilon()) {
            return 0.0f;
        }
        return dot_prod / denominator;
    }

    template<typename T>
    Tensor<T, 3> resizeBilinear(const Tensor<T, 3>& src, size_t target_h, size_t target_w) {
        const size_t* in_shape = src.getShape();
        size_t in_h = in_shape[0];
        size_t in_w = in_shape[1];
        size_t channels = in_shape[2];

        Tensor<T, 3> out({target_h, target_w, channels});

        // Scaling ratios mapping output coordinates back to input coordinates
        float h_ratio = (target_h > 1) ? static_cast<float>(in_h - 1) / (target_h - 1) : 0.0f;
        float w_ratio = (target_w > 1) ? static_cast<float>(in_w - 1) / (target_w - 1) : 0.0f;

        for (size_t y_out = 0; y_out < target_h; ++y_out) {
            float y_in = y_out * h_ratio;
            size_t y0 = static_cast<size_t>(y_in);
            size_t y1 = std::min(y0 + 1, in_h - 1);
            float y_weight = y_in - y0;

            for (size_t x_out = 0; x_out < target_w; ++x_out) {
                float x_in = x_out * w_ratio;
                size_t x0 = static_cast<size_t>(x_in);
                size_t x1 = std::min(x0 + 1, in_w - 1);
                float x_weight = x_in - x0;

                // Interpolate across all color channels
                for (size_t c = 0; c < channels; ++c) {
                    // Four neighboring pixel corners
                    T top_left     = src({y0, x0, c});
                    T top_right    = src({y0, x1, c});
                    T bottom_left  = src({y1, x0, c});
                    T bottom_right = src({y1, x1, c});

                    // Horizontal interpolations
                    float top  = top_left + x_weight * (top_right - top_left);
                    float bottom = bottom_left + x_weight * (bottom_right - bottom_left);

                    // Vertical interpolation
                    float interpolated = top + y_weight * (bottom - top);

                    out({y_out, x_out, c}) = static_cast<T>(interpolated);
                }
            }
        }

        return out;
    }

} // namespace linalg

#endif