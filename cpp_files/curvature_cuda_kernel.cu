#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>


#define NumTri 220 
#define NumTop 96 

__constant__ float eps=1e-6;

__constant__ float thres=1e-4;

__constant__ int acceptTopologyWithFlip[2][96]={ {1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 25, 31, 32, 34, 35, 38, 47, 48, 49, 50, 51, 55, 59, 63, 64, 68, 70, 76, 79, 96, 98, 100, 102, 103, 110, 111, 112, 115, 118, 119, 127, 0, 255, 128, 136, 137, 140, 143, 144, 145, 152, 153, 155, 157, 159, 176, 179, 185, 187, 191, 192, 196, 200, 204, 205, 206, 207, 208, 217, 220, 221, 223, 224, 230, 236, 238, 239, 240, 241, 242, 243, 244, 246, 247, 248, 249, 251, 252, 253, 254},
				    {1, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 3, 2, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 1, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 2, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 1, 3, 3, 3, 2, 1, 2, 3, 3, 2, 3, 2, 1, 3, 2, 1, 2, 1, 1}};

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
  vertices[0] = 0.5-offset[0       + (x+1)*H*D + (y+1)*D + z   ]; 
  vertices[1] = 1.0; 
  vertices[2] = 0.0; 

  vertices[3] = 1.0; 
  vertices[4] = 0.5-offset[1*W*H*D + (x+1)*H*D + (y+1)*D + z   ]; 
  vertices[5] = 0.0; 

  vertices[6] = 0.5-offset[0       + (x+1)*H*D + (y  )*D + z   ]; 
  vertices[7] = 0.0; 
  vertices[8] = 0.0; 

  vertices[9] = 0.0; 
  vertices[10] = 0.5-offset[1*W*H*D + (x  )*H*D + (y+1)*D + z   ]; 
  vertices[11] = 0.0; 

  vertices[12] = 0.5-offset[0       + (x+1)*H*D + (y+1)*D + z+1 ]; 
  vertices[13] = 1.0; 
  vertices[14] = 1.0; 

  vertices[15] = 1.0; 
  vertices[16] = 0.5-offset[1*W*H*D + (x+1)*H*D + (y+1)*D + z+1 ]; 
  vertices[17] = 1.0; 

  vertices[18] = 0.5-offset[0       + (x+1)*H*D + (y  )*D + z+1 ]; 
  vertices[19] = 0.0; 
  vertices[20] = 1.0; 

  vertices[21] = 0.0; 
  vertices[22] = 0.5-offset[1*W*H*D + (x  )*H*D + (y+1)*D + z+1 ]; 
  vertices[23] = 1.0; 

  vertices[24] = 0.0; 
  vertices[25] = 1.0; 
  vertices[26] = 0.5-offset[2*W*H*D + (x  )*H*D + (y+1)*D + z+1 ]; 

  vertices[27] = 1.0; 
  vertices[28] = 1.0; 
  vertices[29] = 0.5-offset[2*W*H*D + (x+1)*H*D + (y+1)*D + z+1 ]; 

  vertices[30] = 1.0; 
  vertices[31] = 0.0; 
  vertices[32] = 0.5-offset[2*W*H*D + (x+1)*H*D + (y  )*D + z+1 ]; 

  vertices[33] = 0.0; 
  vertices[34] = 0.0; 
  vertices[35] = 0.5-offset[2*W*H*D + (x  )*H*D + (y  )*D + z+1 ]; 
}


__device__ void intersection(const int *array1, const int *array2, int *out){

    int count = 0;

    for (int i=0; i<3; i++){
        out[i] = -1;
    }
    

    for (int i=0; i<4; i++){
	for (int j=0; j<3; j++){
	    if (array2[j]==array1[i]){
		out[count] = array1[i]; 
		count ++;
	    }
	}
    }
}

__device__ void get_vertices_on_face(const int r, int *row){
    int vertices_on_location[6][4] = { {5, 9, 1, 10},
                                   {7, 8, 3, 11},
                                   {4, 9, 0, 8},
                                   {6, 10, 2, 11},
                                   {4, 5, 6, 7},
                                   {0, 1, 2, 3} };
    for (int i=0; i<4; i++){
      row[i]=vertices_on_location[r][i];
    }
}


__device__ void dn_dpb( const float *vertices, const int a, const int c, const float *dn, float *db ){

  float d3 = vertices[c*3+2] - vertices[a*3+2];
  float d2 = vertices[c*3+1] - vertices[a*3+1];
  float d1 = vertices[c*3+0] - vertices[a*3+0];

  db[0] =             d3*dn[1] - d2*dn[2];
  db[1] = -d3*dn[0]            + d1*dn[2];
  db[2] =  d2*dn[0] - d1*dn[1]           ;

}

__device__ void dn_dpc( const float *vertices, const int a, const int b, const float *dn, float *dc ){

  float d3 = vertices[b*3+2] - vertices[a*3+2];
  float d2 = vertices[b*3+1] - vertices[a*3+1];
  float d1 = vertices[b*3+0] - vertices[a*3+0];

  dc[0] =           - d3*dn[1] + d2*dn[2];
  dc[1] =  d3*dn[0]            - d1*dn[2];
  dc[2] = -d2*dn[0] + d1*dn[1]           ;

}

__device__ void dn_dpa( const float *vertices, const int b, const int c, const float *dn, float *da ){

  float d3 = vertices[b*3+2] - vertices[c*3+2];
  float d2 = vertices[b*3+1] - vertices[c*3+1];
  float d1 = vertices[b*3+0] - vertices[c*3+0];

  da[0] =             d3*dn[1] - d2*dn[2];
  da[1] = -d3*dn[0]            + d1*dn[2];
  da[2] =  d2*dn[0] - d1*dn[1]           ;
}

__device__ void offset_to_normals(const float *offset, const int W, const int H, const int D, const int i_, const int j_, const int k_, const int location, float *normal, float *length){

  float vertices[12*3];
  offset_to_vertices_cuda(offset, W, H, D, i_, j_, k_, vertices);


  int vertices_on_face[4];
  get_vertices_on_face(location, vertices_on_face); 

  int tri_cnt = 0;
  for (int i = 0; i < NumTop; i++){
      int top_ind = acceptTopologyWithFlip[0][i]; 
      int num_triangle = acceptTopologyWithFlip[1][i];
      for (int tri_ind = 0; tri_ind<num_triangle; tri_ind++){
	  int triangle[3] = {triTable[top_ind][tri_ind*3], triTable[top_ind][tri_ind*3+1], triTable[top_ind][tri_ind*3+2]};
	  
          int inter_ind[3];
	  intersection(vertices_on_face, triangle, inter_ind);


	  if (location>5 ||  (location <=5 && inter_ind[0]>-1 && inter_ind[1]>-1 && inter_ind[2]==-1) ){
	
	    int a, b, c;
	    a = triangle[0];
	    b = triangle[1];
	    c = triangle[2];
		
	    float vec1[3] = { vertices[b*3+0] - vertices[a*3+0], 
		    	      vertices[b*3+1] - vertices[a*3+1],
		    	      vertices[b*3+2] - vertices[a*3+2] };
	    float vec2[3] = { vertices[c*3+0] - vertices[a*3+0], 
		    	      vertices[c*3+1] - vertices[a*3+1],
		    	      vertices[c*3+2] - vertices[a*3+2] };

	    float cross[3] = { vec1[1]*vec2[2] - vec1[2]*vec2[1],
	    		       vec1[2]*vec2[0] - vec1[0]*vec2[2],
	    		       vec1[0]*vec2[1] - vec1[1]*vec2[0] };

	    float l2 = sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
	    if (l2<eps) { l2=eps; }
	    
	    normal[tri_cnt*3 + 0] = cross[0]/l2;
	    normal[tri_cnt*3 + 1] = cross[1]/l2;
	    normal[tri_cnt*3 + 2] = cross[2]/l2;

	    length[tri_cnt] = l2;

	  }
	  else{
	    normal[tri_cnt*3 + 0] = 1.0;
	    normal[tri_cnt*3 + 1] = 1.0;
	    normal[tri_cnt*3 + 2] = 1.0;
	  }

	  tri_cnt ++;
      }
  }
}


__device__ void grad_normal_to_offset(float *grad_offset, const float *grad_normal, const float *offset, const int W, const int H, const int D, const int i_, const int j_, int k_,  const int location){

  float vertices[12*3];
  offset_to_vertices_cuda(offset, W, H, D, i_, j_, k_, vertices);


  int vertices_on_face[4];
  get_vertices_on_face(location, vertices_on_face); 


  int tri_cnt = 0;
  for (int i = 0; i < NumTop; i++){
      int top_ind = acceptTopologyWithFlip[0][i]; 
      int num_triangle = acceptTopologyWithFlip[1][i];
      for (int tri_ind = 0; tri_ind<num_triangle; tri_ind++){

	  float grad_tri[3] = {grad_normal[tri_cnt*3 + 0], grad_normal[tri_cnt*3 +1], grad_normal[tri_cnt*3 + 2]};

	  int triangle[3] = {triTable[top_ind][tri_ind*3], triTable[top_ind][tri_ind*3+1], triTable[top_ind][tri_ind*3+2]};

    int inter_ind[3];
	  intersection(vertices_on_face, triangle, inter_ind);


	  if (location>5 ||  (location <=5 && inter_ind[0]>-1 && inter_ind[1]>-1 && inter_ind[2]==-1) ){

	    int a, b, c;
	    a = triangle[0];
	    b = triangle[1];
	    c = triangle[2];

	    float da[3];
  	    dn_dpa(vertices, b, c, grad_tri, da);

	    atomicAdd( &grad_offset[ vertices_to_offset[a][0]*W*H*D + 
	    		            (vertices_to_offset[a][1]+i_)*H*D + 
  	    	                    (vertices_to_offset[a][2]+j_)*D +
  	    		             vertices_to_offset[a][3]+k_], 
		       da[vertices_to_offset[a][0]] );

	    float db[3];
  	    dn_dpb(vertices, a, c, grad_tri, db);

	    atomicAdd( &grad_offset[ vertices_to_offset[b][0]*W*H*D + 
	    		            (vertices_to_offset[b][1]+i_)*H*D + 
  	    	                    (vertices_to_offset[b][2]+j_)*D +
  	    		             vertices_to_offset[b][3]+k_], 
		       db[vertices_to_offset[b][0]] );

	    float dc[3];
  	    dn_dpc(vertices, a, b, grad_tri, dc);

	    atomicAdd( &grad_offset[ vertices_to_offset[c][0]*W*H*D + 
	    		            (vertices_to_offset[c][1]+i_)*H*D + 
  	    	                    (vertices_to_offset[c][2]+j_)*D +
  	    		             vertices_to_offset[c][3]+k_], 
		       dc[vertices_to_offset[c][0]] );

	  }

	  tri_cnt++;
      }
  }
}

__device__ void grad_normalized_to_normal(float *grad_normal, const float *normal, const float l){

  float orig_normal[3] = {normal[0]*l, normal[1]*l, normal[2]*l};
  
  float l3 = l*l*l;
  float g00 = (orig_normal[1]*orig_normal[1] + orig_normal[2]*orig_normal[2])/l3;
  float g01 = -orig_normal[0]*orig_normal[1]/l3; 
  float g02 = -orig_normal[0]*orig_normal[2]/l3; 
  float g10 = g01; 
  float g11 = (orig_normal[0]*orig_normal[0] + orig_normal[2]*orig_normal[2])/l3;
  float g12 = -orig_normal[1]*orig_normal[2]/l3; 
  float g20 = g02;
  float g21 = g12;
  float g22 = (orig_normal[0]*orig_normal[0] + orig_normal[1]*orig_normal[1])/l3;

  float g1 = grad_normal[0]*g00 + grad_normal[1]*g01 + grad_normal[2]*g02;
  float g2 = grad_normal[0]*g10 + grad_normal[1]*g11 + grad_normal[2]*g12;
  float g3 = grad_normal[0]*g20 + grad_normal[1]*g21 + grad_normal[2]*g22;

  grad_normal[0] = g1;
  grad_normal[1] = g2;
  grad_normal[2] = g3;
}


__global__ void pairwise_loss(const float *offset, const float *topology, const float *mask, float *loss, const int direction){

  int i1 = blockIdx.x;
  int j1 = blockIdx.y;
  int k1 = threadIdx.x;

  int W = gridDim.x;
  int H = gridDim.y;
  int D = blockDim.x;

  int T = NumTri;
  
  int i2=0, j2=0, k2=0, ind1=0, ind2=0;
  if (direction==0){
       if (i1==W-1) return;
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+H*D;
       i2 = i1+1;
       j2 = j1;
       k2 = k1;
  }
  else if (direction==1){
       if (j1==H-1) return;
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+H;
       i2 = i1;
       j2 = j1+1;
       k2 = k1;
  }
  else if (direction==2){
       if (k1==D-1) return;
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+1;
       i2 = i1;
       j2 = j1;
       k2 = k1+1;
  }
  else if (direction==3){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1;
       i2 = i1;
       j2 = j1;
       k2 = k1;
  }
  
  float norm1[NumTri*3];
  float norm2[NumTri*3];
  float length1[NumTri];
  float length2[NumTri];
  offset_to_normals(offset, W+1, H+1, D+1, i1, j1, k1, direction*2, norm1, length1);
  offset_to_normals(offset, W+1, H+1, D+1, i2, j2, k2, direction*2+1, norm2, length2);


  float loss_=0;

  for (int ti=0; ti<T; ti++){

    for (int tj=0; tj<T; tj++){

       float conn_ij = mask[ti*T + tj];
       if (conn_ij<eps) continue;

       float pi = topology[ind1*T+ti];
       float pj = topology[ind2*T+tj];
       float p_ij = pi*pj;

       float diff_norm0 = norm1[ti*3 + 0]-norm2[tj*3 + 0];
       float diff_norm1 = norm1[ti*3 + 1]-norm2[tj*3 + 1];
       float diff_norm2 = norm1[ti*3 + 2]-norm2[tj*3 + 2];

       float loss_tmp = conn_ij * p_ij * (diff_norm0*diff_norm0 + diff_norm1*diff_norm1 + diff_norm2*diff_norm2);
       loss_ += loss_tmp;

    }
  }

  loss[ind1] = loss_;

  __syncthreads();
 
}


__global__ void pairwise_grad(const float *offset, const float *topology, float *grad_offset, const float *mask, const int direction){

  int i1 = blockIdx.x;
  int j1 = blockIdx.y;
  int k1 = threadIdx.x;

  int W = gridDim.x;
  int H = gridDim.y;
  int D = blockDim.x;

  int T = NumTri;
  
  int i2=0, j2=0, k2=0, ind1=0, ind2=0;
  if (direction==0){
       if (i1==W-1) return;
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+H*D;
       i2 = i1+1;
       j2 = j1;
       k2 = k1;
  }
  else if (direction==1){
       if (j1==H-1) return;
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+H;
       i2 = i1;
       j2 = j1+1;
       k2 = k1;
  }
  else if (direction==2){
       if (k1==D-1) return;
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1+1;
       i2 = i1;
       j2 = j1;
       k2 = k1+1;
  }
  else if (direction==3){
       ind1 = i1*H*D + j1*H + k1;
       ind2 = ind1;
       i2 = i1;
       j2 = j1;
       k2 = k1;
  }
  
  float norm1[NumTri*3];
  float norm2[NumTri*3];
  float length1[NumTri];
  float length2[NumTri];
  offset_to_normals(offset, W+1, H+1, D+1, i1, j1, k1, direction*2, norm1, length1);
  offset_to_normals(offset, W+1, H+1, D+1, i2, j2, k2, direction*2+1, norm2, length2);

  float grad_norm1[NumTri*3]={0};
  float grad_norm2[NumTri*3]={0};

  for (int ti=0; ti<T; ti++){

    for (int tj=0; tj<T; tj++){

       float conn_ij = mask[ti*T + tj];
       if (conn_ij<eps) continue;

       float pi = topology[ind1*T+ti];
       float pj = topology[ind2*T+tj];
       float p_ij = pi*pj;

       float grad_norm1_[3] = {-2*conn_ij*p_ij*norm2[tj*3 + 0], -2*conn_ij*p_ij*norm2[tj*3 + 1], -2*conn_ij*p_ij*norm2[tj*3 + 2]};
       float grad_norm2_[3] = {-2*conn_ij*p_ij*norm1[ti*3 + 0], -2*conn_ij*p_ij*norm1[ti*3 + 1], -2*conn_ij*p_ij*norm1[ti*3 + 2]};

       float norm1_[3] = {norm1[ti*3 + 0], norm1[ti*3 + 1], norm1[ti*3 + 2]};
       float norm2_[3] = {norm2[tj*3 + 0], norm2[tj*3 + 1], norm2[tj*3 + 2]};
       grad_normalized_to_normal(grad_norm1_, norm1_, length1[ti]);
       grad_normalized_to_normal(grad_norm2_, norm2_, length2[tj]);
       grad_norm1[ti*3 + 0] += grad_norm1_[0]; 
       grad_norm1[ti*3 + 1] += grad_norm1_[1];
       grad_norm1[ti*3 + 2] += grad_norm1_[2];

       grad_norm2[tj*3 + 0] += grad_norm2_[0];
       grad_norm2[tj*3 + 1] += grad_norm2_[1];
       grad_norm2[tj*3 + 2] += grad_norm2_[2];

    }
  }

  grad_normal_to_offset(grad_offset, grad_norm1, offset, W+1, H+1, D+1, i1, j1, k1, direction*2);
  grad_normal_to_offset(grad_offset, grad_norm2, offset, W+1, H+1, D+1, i2, j2, k2, direction*2 + 1);

}

void curvature_cuda_forward(
    torch::Tensor offset,
    torch::Tensor topology,
    torch::Tensor xTable,
    torch::Tensor yTable,
    torch::Tensor zTable,
    torch::Tensor innerTable,
    torch::Tensor xLoss,
    torch::Tensor yLoss,
    torch::Tensor zLoss,
    torch::Tensor innerLoss,
    torch::Tensor loss){

  int W = offset.size(1)-1;
  int H = offset.size(2)-1;
  int D = offset.size(3)-1;

  dim3 dimGrid(W, H, 1);
  dim3 dimBlock(D, 1, 1);

  float loss_ = 0.0;

  pairwise_loss<<<dimGrid, dimBlock>>>(
            offset.data_ptr<float>(),
            topology.data_ptr<float>(),
            xTable.data_ptr<float>(),
            xLoss.data_ptr<float>(),
            0);
  float lossx = xLoss.sum().item<float>();
  loss_ += lossx;

  pairwise_loss<<<dimGrid, dimBlock>>>(
            offset.data_ptr<float>(),
            topology.data_ptr<float>(),
            yTable.data_ptr<float>(),
            yLoss.data_ptr<float>(),
            0);
  float lossy = yLoss.sum().item<float>();
  loss_ += lossy;

  pairwise_loss<<<dimGrid, dimBlock>>>(
            offset.data_ptr<float>(),
            topology.data_ptr<float>(),
            zTable.data_ptr<float>(),
            zLoss.data_ptr<float>(),
            0);
  float lossz = zLoss.sum().item<float>();
  loss_ += lossz;

  pairwise_loss<<<dimGrid, dimBlock>>>(
            offset.data_ptr<float>(),
            topology.data_ptr<float>(),
            innerTable.data_ptr<float>(),
            innerLoss.data_ptr<float>(),
            3);
  float lossiner = innerLoss.sum().item<float>();
  loss_ += lossiner;

  loss[0] = loss_;
  
}

void curvature_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor offset,
    torch::Tensor topology,
    torch::Tensor xTable, 
    torch::Tensor yTable,
    torch::Tensor zTable,
    torch::Tensor innerTable,
    torch::Tensor grad_offset){

  int W = offset.size(1) - 1;
  int H = offset.size(2) - 1;
  int D = offset.size(3) - 1;

  dim3 dimGrid(W, H, 1);
  dim3 dimBlock(D, 1, 1);

  pairwise_grad<<<dimGrid, dimBlock>>>(
        offset.data_ptr<float>(),
        topology.data_ptr<float>(),
        grad_offset.data_ptr<float>(),
        xTable.data_ptr<float>(),
        0);
        
  pairwise_grad<<<dimGrid, dimBlock>>>(
        offset.data_ptr<float>(),
        topology.data_ptr<float>(),
        grad_offset.data_ptr<float>(),
        yTable.data_ptr<float>(),
        1);

  pairwise_grad<<<dimGrid, dimBlock>>>(
        offset.data_ptr<float>(),
        topology.data_ptr<float>(),
        grad_offset.data_ptr<float>(),
        zTable.data_ptr<float>(),
        2);

  pairwise_grad<<<dimGrid, dimBlock>>>(
        offset.data_ptr<float>(),
        topology.data_ptr<float>(),
        grad_offset.data_ptr<float>(),
        innerTable.data_ptr<float>(),
        3);

 }