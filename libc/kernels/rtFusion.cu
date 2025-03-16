#include "fusion.cuh"
#include <cuda_runtime.h>

//      Core 
#define BLOCK_X 16                // Optimal for memory coalescing
#define BLOCK_Y 16                // 16x16 = 256 threads/block
#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

//      Tensors
#define IN_W (in_w)
#define IN_H (in_h)
#define OUT_SIZE (out_size)

// Boundary clamping
#define CLAMP(v, min, max) ((v) < (min) ? (min) : ((v) > (max) ? (max) : (v)))

//   Precision
#ifdef USE_FP16
#include <cuda_fp16.h>
#define OUTPUT_TYPE half
#else
#define OUTPUT_TYPE float
#endif

//  Indexing Macros 
#define INPUT_INDEX(y, x) ((y) * IN_W + (x))
#define OUT_IDX(ch, y, x) ((ch) * OUT_SIZE * OUT_SIZE + (y) * OUT_SIZE + (x))

//  Bilinear Interpolation
#define BILINEAR_INTERP(v00, v01, v10, v11, a, b) \
    ((1-a)*(1-b)*(v00) + a*(1-b)*(v01) + \
     (1-a)*b*(v10) + a*b*(v11))

__global__ void rtKerr(uchar3* input, OUTPUT_TYPE* output,
                      int in_w, int in_h, int out_size) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= OUT_SIZE || y >= OUT_SIZE) return;

    // Coordinates
    const float fx = x * (IN_W - 1) / (float)(OUT_SIZE - 1);
    const float fy = y * (IN_H - 1) / (float)(OUT_SIZE - 1);
    
    const int x0 = (int)fx;
    const int y0 = (int)fy;
    const int x1 = CLAMP(x0 + 1, 0, IN_W - 1);
    const int y1 = CLAMP(y0 + 1, 0, IN_H - 1);

    const float a = fx - x0;
    const float b = fy - y0;

    // Pixels
    const uchar3 p00 = input[INPUT_INDEX(y0, x0)];
    const uchar3 p01 = input[INPUT_INDEX(y0, x1)];
    const uchar3 p10 = input[INPUT_INDEX(y1, x0)];
    const uchar3 p11 = input[INPUT_INDEX(y1, x1)];

    // Channel processing
    const float red = BILINEAR_INTERP(p00.z, p01.z, p10.z, p11.z, a, b);
    const float green = BILINEAR_INTERP(p00.y, p01.y, p10.y, p11.y, a, b);
    const float blue = BILINEAR_INTERP(p00.x, p01.x, p10.x, p11.x, a, b);

    // Normalization and storage
    const float scale = 1.0f / 255.0f;
    output[OUT_IDX(0, y, x)] = red * scale;    // R
    output[OUT_IDX(1, y, x)] = green * scale;  // G
    output[OUT_IDX(2, y, x)] = blue * scale;   // B
}
