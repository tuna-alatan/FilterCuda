#pragma once
#include <cuda_runtime.h>

class Pixel {
public:
	unsigned char p[3];
public:
	__host__ __device__ Pixel() : p{ 0, 0, 0 } {}
	__host__ __device__ Pixel(unsigned char r, unsigned char g, unsigned char b) : p{ r, g, b } {}

	__host__ __device__ unsigned char r() const { return p[0]; };
	__host__ __device__ unsigned char g() const { return p[1]; };
	__host__ __device__ unsigned char b() const { return p[2]; };

	__host__ __device__ unsigned char operator[](int i) const { return p[i]; }
	__host__ __device__ unsigned char& operator[](int i) { return p[i]; }
};
