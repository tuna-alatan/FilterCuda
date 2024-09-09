#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "pixel.h"

#define KERNEL_SIZE 15
#define OFFSET (KERNEL_SIZE / 2)
#define BLOCK_SIZE 32
#define TILE_SIZE (BLOCK_SIZE - 2 * OFFSET) 
#define SHEM_SIZE (BLOCK_SIZE * BLOCK_SIZE)


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void blur_filter_shared_memory(int w, int h, int c, Pixel* input, unsigned char* output, int kernel_size, int* low_pass_filter) {
    int offset = kernel_size / 2;

    __shared__ Pixel sBuffer[SHEM_SIZE];

    int x_i = TILE_SIZE * blockIdx.x + threadIdx.x - offset;
    int y_i = TILE_SIZE * blockIdx.y + threadIdx.y - offset;
    
    int m = x_i * c;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    int s_index = ty * dim + tx;
    int index1 = y_i * w + x_i;

    if (x_i >= w || y_i  >= h) return;

    if ((x_i >= 0) && (x_i < w) && (y_i >= 0) && (y_i < h))
        sBuffer[s_index] = input[index1];
    else
        sBuffer[s_index] = Pixel(0, 0, 0);
        
    __syncthreads();

    if ((tx >= offset) && (tx < (BLOCK_SIZE - offset)) && (ty >= offset) && (ty < (BLOCK_SIZE - offset))) {
        float sum_r = 0.0;
        float sum_g = 0.0;
        float sum_b = 0.0;

        for (int k = -offset; k <= offset; k++) {
            for (int l = -offset; l <= offset; l++) {
                sum_r += sBuffer[(ty + k) * dim + tx + l].r() * low_pass_filter[(k + offset) * kernel_size + (l + offset)];
                sum_g += sBuffer[(ty + k) * dim + tx + l].g() * low_pass_filter[(k + offset) * kernel_size + (l + offset)];
                sum_b += sBuffer[(ty + k) * dim + tx + l].b() * low_pass_filter[(k + offset) * kernel_size + (l + offset)];
            }
        }

        unsigned char r = sum_r / (kernel_size * kernel_size);
        unsigned char g = sum_g / (kernel_size * kernel_size);
        unsigned char b = sum_b / (kernel_size * kernel_size);

        output[y_i * w * c + m] = r;
        output[y_i * w * c + m + 1] = g;
        output[y_i * w * c + m + 2] = b;
    }
    else {

    }
}

__global__ void blur_filter(int w, int h, int c, Pixel* input, unsigned char* output, int kernel_size, int *low_pass_filter) {
    int offset = (kernel_size - 1) / 2;

    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int m = j * c;

    if (i >= h || j >= w) return;

    float sum_r = 0.0;
    float sum_g = 0.0;
    float sum_b = 0.0;

    for (int k = -offset; k <= offset; k++) {
        for (int l = -offset; l <= offset; l++) {
            int index_x;
            int index_y;

            // Mirroring the pixels if exceed the boundries
            (j + l) < w ? index_x = j + l : index_x = j - ((j + l) % w);
            (i + k) < h ? index_y = i + k : index_y = i - ((i + k) % h);

            sum_r += (float)(input[abs(index_y) * w + abs(index_x)].r() * low_pass_filter[(k + offset) * kernel_size + l + offset]);
            sum_g += (float)(input[abs(index_y) * w + abs(index_x)].g() * low_pass_filter[(k + offset) * kernel_size + l + offset]);
            sum_b += (float)(input[abs(index_y) * w + abs(index_x)].b() * low_pass_filter[(k + offset) * kernel_size + l + offset]);
        }
    }

    unsigned char r = sum_r / (kernel_size * kernel_size);
    unsigned char g = sum_g / (kernel_size * kernel_size);
    unsigned char b = sum_b / (kernel_size * kernel_size);

    output[i * w * c + m] = r;
    output[i * w * c + m + 1] = g;
    output[i * w * c + m + 2] = b;
}

int main() {
    int w, h, c;

    // Loading the image
    unsigned char* img_loaded = stbi_load("example.jpg", &w, &h, &c, 3);

    // Checking if loading process is successful
    if (img_loaded == NULL) {
        printf("Couldn't load image");
        return;
    }

    int n_pixels = w * h;
    Pixel* img_in;
    cudaMallocManaged(&img_in, n_pixels * sizeof(Pixel));
    
    unsigned char* img_out;
    cudaMallocManaged((void**)&img_out, n_pixels * c * sizeof(unsigned char));
    cudaMemset(img_out, 0, n_pixels * c * sizeof(unsigned char));

    for (int i = 0; i < h; i++) {
        for (int j = 0, k = 0; k < w; j += c, k++) {
            unsigned char r = img_loaded[j + (i * w * c)];
            unsigned char g = img_loaded[j + 1 + (i * w * c)];
            unsigned char b = img_loaded[j + 2 + (i * w * c)];

            img_in[i * w + k] = Pixel(r, g, b);
        }
    }

    stbi_image_free(img_loaded);

    size_t lpf_size = KERNEL_SIZE * KERNEL_SIZE * sizeof(int);

    int* low_pass_filter;
    cudaMallocManaged(&low_pass_filter, lpf_size);
    for (int i = 0; i < KERNEL_SIZE * KERNEL_SIZE; i++) {
        low_pass_filter[i] = 1;
    }

    // Timing the function
    auto start = std::chrono::high_resolution_clock::now();

    // Cuda Stuff
    int tx = 32;
    int ty = 32;
    
    dim3 blocks(w / TILE_SIZE + 1, h / TILE_SIZE + 1);
    dim3 threads(tx, ty);

    blur_filter_shared_memory <<<blocks, threads >>> (w, h, c, img_in, img_out, KERNEL_SIZE, low_pass_filter);
    //blur_filter <<<blocks, threads>>> (w, h, c, img_in, img_out, KERNEL_SIZE, low_pass_filter);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Eplased: " << duration.count() << " milliseconds" << "\n";

    stbi_write_jpg("out.jpg", w, h, c, img_out, 100);

    cudaFree(img_in);
    cudaFree(img_out);
    cudaFree(low_pass_filter);


    return 0;
}