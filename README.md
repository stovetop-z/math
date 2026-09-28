# Math

A small C++ project with a fixed-rank, row-major `Tensor<T, N>` type and a collection of linear algebra and statistics helpers. The current executable demonstrates those pieces by loading two images, resizing them to a common size, and reporting their means and cosine similarity.

## Contents

- `math/tensor.h` and `math/tensor.cc` define the tensor type, including shape and length access, indexed access, and copying.
- `math/linalg.h` provides tensor addition and subtraction, min/max, mean, variance, standard deviation, normalization, dot product, cosine similarity, and bilinear resize for rank-three tensors.
- `main.cc` is the image comparison example. It uses the bundled `stb_image` and `stb_image_write` headers (the latter is included with its implementation enabled, though the example does not write an image).
- `images/` contains sample input images.

## Build

Build from the repository root with a C++11-compatible compiler:

```sh
g++ -std=c++17 -O2 -o image_test main.cc
```

The tensor implementation is included by `math/tensor.h`, so `main.cc` is the only translation unit needed for this example. No separate library or package installation is required; the image headers are included in the repository.

## Run

The program defaults to `images/me.png` and `images/me_copy.png`:

```sh
./image_test
```

You can pass two image paths instead:

```sh
./image_test path/to/first.png path/to/second.png
```

Both images are loaded as RGB, converted to float values in `[0, 1]`, and resized to `255 × 255`. The program prints each resized image's mean and their cosine similarity. It exits with an error if either input cannot be loaded.

## Using tensors

Include `math/linalg.h` to use `Tensor` and the helpers. Tensor dimensions are supplied as a fixed-size array; a rank-three image uses `[height, width, channels]` order.

```cpp
#include "math/linalg.h"

int main() {
    Tensor<float, 2> values({2, 3});
    values[0] = 1.0f;
    values({1, 2}) = 6.0f;

    float average = linalg::mean(values);
    return average > 0.0f ? 0 : 1;
}
```

The tensor stores elements contiguously in row-major order. `operator[]` accesses the flattened storage, while `operator()` and `at()` access an element using an index for each dimension.
