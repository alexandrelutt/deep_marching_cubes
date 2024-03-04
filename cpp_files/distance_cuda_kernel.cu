#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__constant__ float eps=1e-8;

__constant__ float grid_size = 1.0;

__constant__ float distance_empty = 0.4;

__constant__ int acceptTopology[2][48] = {{1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 25, 31, 32, 34, 35, 38, 47, 48, 49, 50, 51, 55, 59, 63, 64, 68, 70, 76, 79, 96, 98, 100, 102, 103, 110, 111, 112, 115, 118, 119, 127, 0},
			{1, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 3, 2, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 1, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 0}};

__constant__ int triTable[256][16] =
{{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

__constant__ int vertices_to_offset[12][4]={ {0, 1, 1, 0}, // #0
				{1, 1, 1, 0}, // #1
				{0, 1, 0, 0}, // #2
				{1, 0, 1, 0}, // #3
				
				{0, 1, 1, 1}, // #4
				{1, 1, 1, 1}, // #5
				{0, 1, 0, 1}, // #6
				{1, 0, 1, 1}, // #7

				{2, 0, 1, 1}, // #8
				{2, 1, 1, 1}, // #9
				{2, 1, 0, 1}, // #10
				{2, 0, 0, 1}}; // #11


__device__ void offset_to_vertices_cuda(const float *offset, const int W, const int H, const int D, const int x, const int y, const int z, float *vertices){
  // #0
  vertices[0 ] = 0.5-offset[0       + (x+1)*H*D + (y+1)*D + z   ]; 
  vertices[1 ] = 1.0; 
  vertices[2 ] = 0.0; 
  // #1
  vertices[3 ] = 1.0; 
  vertices[4 ] = 0.5-offset[1*W*H*D + (x+1)*H*D + (y+1)*D + z   ]; 
  vertices[5 ] = 0.0; 
  // #2
  vertices[6 ] = 0.5-offset[0       + (x+1)*H*D + (y  )*D + z   ]; 
  vertices[7 ] = 0.0; 
  vertices[8 ] = 0.0; 
  // #3
  vertices[9 ] = 0.0; 
  vertices[10] = 0.5-offset[1*W*H*D + (x  )*H*D + (y+1)*D + z   ]; 
  vertices[11] = 0.0; 

  // #4
  vertices[12] = 0.5-offset[0       + (x+1)*H*D + (y+1)*D + z+1 ]; 
  vertices[13] = 1.0; 
  vertices[14] = 1.0; 
  // #5
  vertices[15] = 1.0; 
  vertices[16] = 0.5-offset[1*W*H*D + (x+1)*H*D + (y+1)*D + z+1 ]; 
  vertices[17] = 1.0; 
  // #6
  vertices[18] = 0.5-offset[0       + (x+1)*H*D + (y  )*D + z+1 ]; 
  vertices[19] = 0.0; 
  vertices[20] = 1.0; 
  // #7
  vertices[21] = 0.0; 
  vertices[22] = 0.5-offset[1*W*H*D + (x  )*H*D + (y+1)*D + z+1 ]; 
  vertices[23] = 1.0; 

  // #8
  vertices[24] = 0.0; 
  vertices[25] = 1.0; 
  vertices[26] = 0.5-offset[2*W*H*D + (x  )*H*D + (y+1)*D + z+1 ]; 
  // #9
  vertices[27] = 1.0; 
  vertices[28] = 1.0; 
  vertices[29] = 0.5-offset[2*W*H*D + (x+1)*H*D + (y+1)*D + z+1 ]; 
  // #10
  vertices[30] = 1.0; 
  vertices[31] = 0.0; 
  vertices[32] = 0.5-offset[2*W*H*D + (x+1)*H*D + (y  )*D + z+1 ]; 
  // #11
  vertices[33] = 0.0; 
  vertices[34] = 0.0; 
  vertices[35] = 0.5-offset[2*W*H*D + (x  )*H*D + (y  )*D + z+1 ]; 
}

__device__ float d_sqrdistance_(float a, float b, float c, float d, float e, float f, float s, float t,
		float d_a, float d_b, float d_c, float d_d, float d_e, float d_f, float d_s, float d_t){
  return d_a*s*s + 2.0*a*d_s*s + 
	 d_c*t*t + 2.0*c*d_t*t + 
	 2.0*d_b*s*t + 2.0*b*d_s*t + 2*b*s*d_t +
	 2.0*d_s*d + 2.0*s*d_d + 
	 2.0*d_e*t + 2.0*e*d_t + d_f;  
}

__device__ float d_s_(float a, float b, float c, float d, float e, 
	float d_a, float d_b, float d_c, float d_d, float d_e,
       	float s_clamp, float t_clamp, float det){
  if (s_clamp==0) return 0;

  if (s_clamp+t_clamp<=1){
    float d_det = d_a*c + a*d_c - 2.0*b*d_b;
    float det2 = det*det;
    if (det2<eps) det2=eps;
    return ((d_b*e + b*d_e - d_c*d - c*d_d)*det -  (b*e-c*d)*d_det ) / ( det2 ); 
  }else if (s_clamp + t_clamp >1 && t_clamp > 0){
    float tmp = b*e - c*d + b*d - a*e;
    return ((d_b*e + b*d_e - d_c*d - c*d_d)*(b*d - a*e) - (b*e-c*d)*(d_b*d + b*d_d - d_a*e - a*d_e) ) / (tmp*tmp);
  }else{
    return 0;
  }
}

__device__ float d_t_(float a, float b, float c, float d, float e, 
	float d_a, float d_b, float d_c, float d_d, float d_e, 
       	float s_clamp, float t_clamp, float det){
  if (t_clamp==0) return 0;

  if (s_clamp+t_clamp<=1){
    float d_det = d_a*c + a*d_c - 2.0*b*d_b;
    float det2 = det*det;
    if (det2<eps) det2=eps;
    return ((d_b*d + b*d_d - d_a*e - a*d_e)*det -  (b*d-a*e)*d_det ) / ( det2 ); 
  }else if (s_clamp + t_clamp >1 && s_clamp > 0){
    float tmp = b*e - c*d + b*d - a*e;
    return ((d_b*d + b*d_d - d_a*e - a*d_e)*(b*e - c*d) - (b*d-a*e)*(d_b*e + b*d_e - d_c*d - c*d_d)) / (tmp * tmp);
  }else{
    return 0;
  }
}

__device__ void grad_triangle_to_offset(const float *grad_triangle, float *grad_offset, const int W, const int H, const int D, const int i, const int j, const int k, const int t, const float count){
  // for triangles in a single toplogy
  for (int tri_ind = 0; tri_ind<acceptTopology[1][t]; tri_ind++){
    // for vertices on the triangle
    for (int vertex_ind = 0; vertex_ind<3; vertex_ind++){

	// every vertex only contributes to the gradient of a single variable on the offset map
	int topology_ind = acceptTopology[0][t];
        int vertex = triTable[topology_ind][tri_ind*3+vertex_ind];

	atomicAdd( &grad_offset[vertices_to_offset[vertex][0]*W*H*D + 
			       (vertices_to_offset[vertex][1]+i)*H*D + 
  		               (vertices_to_offset[vertex][2]+j)*D +
  			        vertices_to_offset[vertex][3]+k], 
		   -grad_triangle[ tri_ind*9 + vertex_ind*3 + vertices_to_offset[vertex][0] ]/count );

	}
  }
}

__device__ float point_triangle_distance_forward(const float *triangle, const float *point)
{
 
  float det, s, t, sqrdistance;

  float B[3] = {triangle[0], triangle[1], triangle[2]}; 
  float E0[3] = {triangle[3]-B[0], triangle[4]-B[1], triangle[5]-B[2]};
  float E1[3] = {triangle[6]-B[0], triangle[7]-B[1], triangle[8]-B[2]};

  float a = E0[0]*E0[0] + E0[1]*E0[1] + E0[2]*E0[2];
  float b = E0[0]*E1[0] + E0[1]*E1[1] + E0[2]*E1[2];
  float c = E1[0]*E1[0] + E1[1]*E1[1] + E1[2]*E1[2];

  float D[3] = {B[0]-point[0], B[1]-point[1], B[2]-point[2]};
  float d = E0[0]*D[0] + E0[1]*D[1] + E0[2]*D[2];
  float e = E1[0]*D[0] + E1[1]*D[1] + E1[2]*D[2];
  float f = D[0]*D[0] + D[1]*D[1] + D[2]*D[2];

  det = a*c - b*b;
  if (det<eps) det=eps;
  s = (b*e - c*d) / det;
  t = (b*d - a*e) / det;

  if (s<0) s=0;
  if (t<0) t=0;
  float norm = s+t;
  if (norm>1){
          s = s/norm;
          t = t/norm;
  }

  sqrdistance = s * ( a*s + b*t + 2.0*d ) + t * ( b*s + c*t + 2.0*e ) + f;
  return sqrdistance;
}

__device__ void point_triangle_distance_backward(const float grad_output_, const float *triangle, const float *point, float *grad_triangle)
{
  float det, s, t;

  float t11, t12, t13, t21, t22, t23, t31, t32, t33;
  float p1, p2, p3;
  t11 = triangle[0];
  t21 = triangle[1];
  t31 = triangle[2];
  t12 = triangle[3];
  t22 = triangle[4];
  t32 = triangle[5];
  t13 = triangle[6];
  t23 = triangle[7];
  t33 = triangle[8];

  p1 = point[0];
  p2 = point[1];
  p3 = point[2];

  float B[3] = {triangle[0], triangle[1], triangle[2]}; 
  float E0[3] = {triangle[3]-B[0], triangle[4]-B[1], triangle[5]-B[2]};
  float E1[3] = {triangle[6]-B[0], triangle[7]-B[1], triangle[8]-B[2]};

  float a = E0[0]*E0[0] + E0[1]*E0[1] + E0[2]*E0[2];
  float b = E0[0]*E1[0] + E0[1]*E1[1] + E0[2]*E1[2];
  float c = E1[0]*E1[0] + E1[1]*E1[1] + E1[2]*E1[2];


  float d_t11,d_t21,d_t31,d_t12,d_t22,d_t32,d_t13,d_t23,d_t33;
  d_t11=d_t21=d_t31=d_t12=d_t22=d_t32=d_t13=d_t23=d_t33 = 0;


  float D[3] = {B[0]-p1, B[1]-p2, B[2]-p3};
  float d = E0[0]*D[0] + E0[1]*D[1] + E0[2]*D[2];
  float e = E1[0]*D[0] + E1[1]*D[1] + E1[2]*D[2];
  float f = D[0]*D[0] + D[1]*D[1] + D[2]*D[2];

  det = a*c - b*b;
  if (det<eps) det=eps;
  s = (b*e - c*d) / det;
  t = (b*d - a*e) / det;

  float d_a, d_b, d_c, d_d, d_e, d_f;
  float s_clamp = s; 
  float t_clamp = t;
  if (s<0) s_clamp=0;
  if (t<0) t_clamp=0;
  float s_norm = s_clamp;
  float t_norm = t_clamp;
  float norm = s_clamp+t_clamp;
  if (norm>1){
          s_norm = s_clamp/norm;
          t_norm = t_clamp/norm;
  }

  // t11
  d_a = 2*t11 - 2*t12; d_b = 2*t11 - t12 - t13; d_c = 2*t11 - 2*t13; d_d = p1 - 2*t11 + t12; d_e = p1 - 2*t11 + t13; d_f = 2*t11 - 2*p1; 
  d_t11 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
  // t21
  d_a = 2*t21 - 2*t22; d_b = 2*t21 - t22 - t23; d_c = 2*t21 - 2*t23; d_d = p2 - 2*t21 + t22; d_e = p2 - 2*t21 + t23; d_f = 2*t21 - 2*p2; 
  d_t21 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
  // t31
  d_a = 2*t31 - 2*t32; d_b = 2*t31 - t32 - t33; d_c = 2*t31 - 2*t33; d_d = p3 - 2*t31 + t32; d_e = p3 - 2*t31 + t33; d_f = 2*t31 - 2*p3; 
  d_t31 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));

  // t12
  d_a = 2*t12 - 2*t11; d_b = t13 - t11; d_c = 0.0; d_d = t11 - p1; d_e = 0.0; d_f = 0.0; 
  d_t12 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
  // t22
  d_a = 2*t22 - 2*t21; d_b = t23 - t21; d_c = 0.0; d_d = t21 - p2; d_e = 0.0; d_f = 0.0; 
  d_t22 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));

  d_a = 2*t32 - 2*t31; d_b = t33 - t31; d_c = 0.0; d_d = t31 - p3; d_e = 0.0; d_f = 0.0; 
  d_t32 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));

  // t13
  d_a = 0.0; d_b = t12 - t11; d_c = 2*t13 - 2*t11; d_d = 0.0; d_e = t11 - p1; d_f = 0.0; 
  d_t13 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
  // t23
  d_a = 0.0; d_b = t22 - t21; d_c = 2*t23 - 2*t21; d_d = 0.0; d_e = t21 - p2; d_f = 0.0; 
  d_t23 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));
  // t33
  d_a = 0.0; d_b = t32 - t31; d_c = 2*t33 - 2*t31; d_d = 0.0; d_e = t31 - p3; d_f = 0.0; 
  d_t33 += grad_output_ * d_sqrdistance_(a,b,c,d,e,f,s_norm,t_norm, d_a,d_b,d_c,d_d,d_e,d_f, d_s_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det), d_t_(a,b,c,d,e, d_a,d_b,d_c,d_d,d_e, s_clamp,t_clamp,det));

  grad_triangle[0] = d_t11;
  grad_triangle[1] = d_t21;
  grad_triangle[2] = d_t31;

  grad_triangle[3] = d_t12;
  grad_triangle[4] = d_t22;
  grad_triangle[5] = d_t32;

  grad_triangle[6] = d_t13;
  grad_triangle[7] = d_t23;
  grad_triangle[8] = d_t33;

}

__global__ void point_toplogy_distance_kernel(const float *offset, const float *points, float *distances, long int *indices, const int n){
  // topology
  int t = threadIdx.x;
  int topology_ind = acceptTopology[0][t];
  int T = blockDim.x + 1;
  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  int Wc = gridDim.x;
  int Hc = gridDim.y;
  int Dc = gridDim.z;
  int ind = i*Hc*Dc + j*Dc + k;
  // offset size, note that we always have 3x(W+1)x(H+1)x(Dx1) offset for WxHxD grid
  int W = Wc + 1;
  int H = Hc + 1;
  int D = Dc + 1;

  // offset_to_vertices
  float vertices[12*3];
  offset_to_vertices_cuda(offset, W, H, D, i, j, k, vertices);

  //float *triangle = offset_to_triangles(offset, i, j, k, t);

  float distance_sum=0.0;
  float count=0;

  for (int p=0; p<n; p++){
     float px = points[p*3+0];
     float py = points[p*3+1];
     float pz = points[p*3+2];
     // if point is inside of the grid
     if (px >= i && px < i+grid_size && py >= j && py < j+grid_size && pz >= k && pz < k+grid_size){
       
       // min distance to a triangle in the same topology
       // also save the min indice for back-propagation
       float min_distance = 10000.0;
       long int min_indice = -1;

       for (int tri_ind = 0; tri_ind<acceptTopology[1][t]; tri_ind++){
         // offset_to_triangles
	 // Note: offset_to_triangles is inside of the loop to avoid dynamically allocate memory, different to cpu version
	 float triangle_single[3*3] = { // v1 
		 			vertices[triTable[topology_ind][tri_ind*3+0]*3 + 0] + float(i),
				        vertices[triTable[topology_ind][tri_ind*3+0]*3 + 1] + float(j),
				        vertices[triTable[topology_ind][tri_ind*3+0]*3 + 2] + float(k), 
					// v2
		 			vertices[triTable[topology_ind][tri_ind*3+1]*3 + 0] + float(i),
				        vertices[triTable[topology_ind][tri_ind*3+1]*3 + 1] + float(j),
				        vertices[triTable[topology_ind][tri_ind*3+1]*3 + 2] + float(k), 
					// v3
		 			vertices[triTable[topology_ind][tri_ind*3+2]*3 + 0] + float(i),
				        vertices[triTable[topology_ind][tri_ind*3+2]*3 + 1] + float(j),
				        vertices[triTable[topology_ind][tri_ind*3+2]*3 + 2] + float(k) }; 
	 float point_single[3] = {px, py, pz};
         float distance_single = point_triangle_distance_forward( triangle_single, point_single );

	 if (distance_single < min_distance){
           min_distance = distance_single;
	   min_indice = tri_ind;
	 }

       }
       indices[p*T + t] = min_indice;
       distance_sum += min_distance;
       count += 1;
     }
  }
  // if the current grid is not empty
  if (count>0) {
    distances[ind*T + t] = distance_sum/count;
  } else {
    distances[ind*T + t] = distance_empty; 
  }

  __syncthreads();

}

__global__ void grad_point_toplogy_distance_kernel(const float *grad_output, const float *offset, const float *points, const long int *indices, float *grad_offset, const int n){

  // topology
  int t = threadIdx.x;
  int T = blockDim.x + 1;
  int topology_ind = acceptTopology[0][t];
  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  int Wc = gridDim.x;
  int Hc = gridDim.y;
  int Dc = gridDim.z;
  int ind = i*Hc*Dc + j*Dc + k;

  int grad_ind = ind*T + t; 
  const float grad_output_element = grad_output[grad_ind];
  //printf("%d %d %d, %d, grad_output_element %f\n", i, j, k, grad_ind, grad_output_element );
  // offset size, note that we always have 3x(W+1)x(H+1)x(Dx1) offset for WxHxD grid
  int W = Wc + 1;
  int H = Hc + 1;
  int D = Dc + 1;

  // offset_to_vertices
  float vertices[12*3];
  offset_to_vertices_cuda(offset, W, H, D, i, j, k, vertices);

  //float *triangle = offset_to_triangles(offset, i, j, k, t);

  float count=0;

  // allocate memory for accumulating the gradients
  // assuming maximum number of triangles for each topology is 4 as in Marching Cubes
  float grad_triangle_all[4*3*3] = {0};


  for (int p=0; p<n; p++){
     float px = points[p*3+0];
     float py = points[p*3+1];
     float pz = points[p*3+2];
     // if point is inside of the grid
     if (px >= float(i) && px < float(i)+grid_size && py >= float(j) && py < float(j)+grid_size && pz >= float(k) && pz < float(k)+grid_size){

       // printf("(%f %f %f) in [%f %f %f]\n", px, py, pz, float(i), float(j), float(k) );
       // printf("grad_output_element %f\n", grad_output_element );
       // printf("grad_output_element index: %d*%d + %d = %d\n", ind, T, t, ind*T+t);
       // only back propagate to the nearest triangle
       int tri_ind = indices[p*T + t];
       if (tri_ind == -1) continue;
       // offset_to_triangles
       // Note: offset_to_triangles is inside of the loop to avoid dynamically allocate memory, different to cpu version
       float triangle_single[3*3] = { // v1 
                    vertices[triTable[topology_ind][tri_ind*3+0]*3 + 0] + float(i),
          		        vertices[triTable[topology_ind][tri_ind*3+0]*3 + 1] + float(j),
          		        vertices[triTable[topology_ind][tri_ind*3+0]*3 + 2] + float(k), 
          			// v2
           			vertices[triTable[topology_ind][tri_ind*3+1]*3 + 0] + float(i),
          		        vertices[triTable[topology_ind][tri_ind*3+1]*3 + 1] + float(j),
          		        vertices[triTable[topology_ind][tri_ind*3+1]*3 + 2] + float(k), 
          			// v3
           			vertices[triTable[topology_ind][tri_ind*3+2]*3 + 0] + float(i),
          		        vertices[triTable[topology_ind][tri_ind*3+2]*3 + 1] + float(j),
          		        vertices[triTable[topology_ind][tri_ind*3+2]*3 + 2] + float(k) }; 
        float point_single[3] = {px, py, pz};
	    float grad_triangle[3*3];
        //point_triangle_distance_backward(grad_output[ind*T + t], triangle_single, point_single, grad_triangle);
        point_triangle_distance_backward(grad_output_element, triangle_single, point_single, grad_triangle);

        // accumulate gradients over all the points for each triangle
            // to reduce times of updating global memory	
        for (int gi=0; gi<9; gi++){
            grad_triangle_all[tri_ind*9 + gi] += grad_triangle[gi];
        }

        count += 1.0;
      }
  }

  if (count<1.0) return;

  // 
  grad_triangle_to_offset(grad_triangle_all, grad_offset, W, H, D, i, j, k, t, count);

}

__global__ void update_empty_topology(float *distances, const int T){
  // cell indices
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = blockIdx.z;
  // cell size
  int Hc = gridDim.y;
  int Dc = gridDim.z;
  int ind = i*Hc*Dc + j*Dc + k;

  int empty=1;
  float max_distance = -1.0;
  for (int t=0; t<T-1; t++){
    float d = distances[ind*T + t];
    if (d!=distance_empty) empty = 0;
    if (d>max_distance) max_distance = d; 
  }

  if (empty==0) {
	  distances[ind*T + T-1] = max_distance*10.0;
  }

  __syncthreads();

}

void dist_cuda_forward(
    torch::Tensor offset,
    torch::Tensor points,
    torch::Tensor distances_full,
    torch::Tensor indices){

    int W, D, H, T;
    W = offset.size(1)-1;
    H = offset.size(2)-1;
    D = offset.size(3)-1;
    T = distances.size(1);

    dim3 dimGrid(W, H, D);
    dim3 dimBlock(T-1, 1, 1);

    int n = points.size(0);

    point_toplogy_distance_kernel<<<dimGrid, dimBlock>>>(
        offset.data_ptr<float>(),
        points.data_ptr<float>(),
        distances_full.data_ptr<float>(),
        indices.data_ptr<long int>(),
        n
    );

    update_empty_topology<<<dimGrid, dimBlock>>>(
        distances_full.data_ptr<float>(),
        T
    );

}

void dist_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor offset,
    torch::Tensor points,
    torch::Tensor indices,
    torch::Tensor grad_offset){

    int W, D, H, T;
    W = offset.size(1)-1;
    H = offset.size(2)-1;
    D = offset.size(3)-1;
    T = grad_output.size(1);

    dim3 dimGrid(W, H, D);
    dim3 dimBlock(T-1, 1, 1);

    int n = points.size(0);

    grad_point_toplogy_distance_kernel<<<dimGrid, dimBlock>>>(
        grad_output.data_ptr<float>(),
        offset.data_ptr<float>(),
        points.data_ptr<float>(),
        indices.data_ptr<long int>(),
        grad_offset.data_ptr<float>(),
        n
    );
}