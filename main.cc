#include "math/linalg.h"

#define STB_IMAGE_IMPLEMENTATION
#include "image_headers/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "image_headers/stb_image_write.h"

#include <iostream>

int main(int argc, char* argv[])
{
    // Use plain const char* to avoid manual dynamic memory allocation
    const char* path1 = "images/me.png";
    const char* path2 = "images/me_copy.png";

    if (argc >= 3)
    {
        path1 = argv[1];
        path2 = argv[2];
    }

    int w1, h1, actual_c1;
    int w2, h2, actual_c2;

    // Force 3 channels (RGB)
    constexpr size_t CHANNELS = 3;
    unsigned char* rp1 = stbi_load(path1, &w1, &h1, &actual_c1, CHANNELS);
    unsigned char* rp2 = stbi_load(path2, &w2, &h2, &actual_c2, CHANNELS);

    if (!rp1)
    {
        std::cerr << "Error: Could not load image from " << path1 << "\n";
        if (rp2) stbi_image_free(rp2);
        return 1;
    }
    if (!rp2)
    {
        std::cerr << "Error: Could not load image from " << path2 << "\n";
        stbi_image_free(rp1);
        return 1;
    }

    // Standard shape: [Height, Width, Channels]
    Tensor<float, 3> image1({static_cast<size_t>(h1), static_cast<size_t>(w1), CHANNELS});
    Tensor<float, 3> image2({static_cast<size_t>(h2), static_cast<size_t>(w2), CHANNELS});

    // Properly cast byte values to float and normalize to [0.0, 1.0]
    for (size_t i = 0; i < image1.getLength(); ++i)
    {
        image1[i] = static_cast<float>(rp1[i]) / 255.0f;
    }
    for (size_t i = 0; i < image2.getLength(); ++i)
    {
        image2[i] = static_cast<float>(rp2[i]) / 255.0f;
    }

    // Free STB buffers now that tensor owns its internal data
    stbi_image_free(rp1);
    stbi_image_free(rp2);

    Tensor<float, 3> t1 = linalg::resizeBilinear(image1, 255, 255);
    Tensor<float, 3> t2 = linalg::resizeBilinear(image2, 255, 255);

    std::cout << path1 << " mean: " << linalg::mean(t1) << "\n";
    std::cout << path2 << " mean: " << linalg::mean(t2) << "\n";
    
    std::cout << "Cosine similarity: " << linalg::cosineSimilarity(t1, t2) << "\n";

    return 0;
}