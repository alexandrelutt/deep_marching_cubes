#include <torch/extension.h>

#include <vector>

static int visTopology[2][140]={{0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 23, 25, 27, 29, 31, 32, 34, 35, 38, 39, 43, 46, 47, 48, 49, 50, 51, 54, 55, 57, 59, 63, 64, 68, 70, 71, 76, 77, 78, 79, 95, 96, 98, 99, 100, 102, 103, 108, 110, 111, 112, 113, 114, 115, 116, 118, 119, 123, 126, 127, 128, 136, 137, 139, 140, 141, 142, 143, 144, 145, 147, 152, 153, 155, 156, 157, 159, 175, 176, 177, 178, 179, 183, 184, 185, 187, 189, 191, 192, 196, 198, 200, 201, 204, 205, 206, 207, 208, 209, 212, 216, 217, 219, 220, 221, 222, 223, 224, 226, 228, 230, 231, 232, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255},
{0, 1, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 3, 2, 1, 2, 3, 4, 3, 4, 4, 3, 1, 2, 3, 3, 4, 4, 4, 3, 2, 3, 3, 2, 4, 3, 4, 3, 2, 1, 2, 3, 4, 3, 4, 4, 3, 4, 2, 3, 4, 3, 2, 3, 4, 3, 2, 3, 4, 4, 3, 4, 3, 2, 4, 4, 1, 1, 2, 3, 4, 3, 4, 4, 3, 2, 3, 4, 3, 2, 3, 4, 3, 2, 4, 3, 4, 4, 3, 4, 4, 3, 2, 4, 1, 2, 3, 4, 3, 4, 2, 3, 3, 2, 3, 4, 4, 4, 3, 4, 3, 2, 4, 1, 3, 4, 4, 3, 4, 4, 3, 4, 2, 1, 2, 3, 3, 2, 3, 4, 2, 1, 3, 2, 4, 1, 2, 1, 1, 0}};

// OFFSET HAS SHAPE 3x(N+1)x(N+1)x(N+1)

torch::Tensor offset_to_vertices(torch::Tensor *offset, int x, int y, int z){
    torch::Tensor vertices = torch::empty({3, 12}); 
    int N = 32;

    vertices[3*0 + 0] = 0.5 - offset[0*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + (y+1)*(N+1) + z];
    vertices[3*0 + 1] = 1
    vertices[3*0 + 2] = 0;

    vertices[3*1 + 0] = 1;
    vertices[3*1 + 1] = 0.5 - offset[1*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + (y+1)*(N+1) + z];
    vertices[3*1 + 2] = 0;

    vertices[3*2 + 0] = 0.5 - offset[0*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + y*(N+1) + z];
    vertices[3*2 + 1] = 0;
    vertices[3*2 + 2] = 0;

    vertices[3*3 + 0] = 0;
    vertices[3*3 + 1] = 0.5 - offset[1*((N+1)*(N+1)*(N+1)) + x*((N+1)*(N+1)) + (y+1)*(N+1) + z];
    vertices[3*3 + 2] = 0;

    vertices[3*4 + 0] = 0.5 - offset[0*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + (y+1)*(N+1) + (z+1)];
    vertices[3*4 + 1] = 1;
    vertices[3*4 + 2] = 1;

    vertices[3*5 + 0] = 1;
    vertices[3*5 + 1] = 0.5 - offset[1*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + (y+1)*(N+1) + (z+1)];
    vertices[3*5 + 2] = 1;

    vertices[3*6 + 0] = 0.5 - offset[0*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + y*(N+1) + (z+1)];
    vertices[3*6 + 1] = 0;
    vertices[3*6 + 2] = 1;

    vertices[3*7 + 0] = 0;
    vertices[3*7 + 1] = 0.5 - offset[1*((N+1)*(N+1)*(N+1)) + x*((N+1)*(N+1)) + (y+1)*(N+1) + (z+1)];
    vertices[3*7 + 2] = 1;

    vertices[3*8 + 0] = 0.5 - offset[0*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + (y+1)*(N+1) + (z+1)];
    vertices[3*8 + 1] = 1;
    vertices[3*8 + 2] = 0.5 - offset[2*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + (y+1)*(N+1) + (z+1)];

    vertices[3*9 + 0] = 1;
    vertices[3*9 + 1] = 1;
    vertices[3*9 + 2] = 0.5 - offset[2*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + (y+1)*(N+1) + (z+1)];

    vertices[3*10 + 0] = 1;
    vertices[3*10 + 1] = 0;
    vertices[3*10 + 2] = 0.5 - offset[2*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + y*(N+1) + (z+1)];

    vertices[3*11 + 0] = 0;
    vertices[3*11 + 1] = 0;
    vertices[3*11 + 2] = 0.5 - offset[2*((N+1)*(N+1)*(N+1)) + (x+1)*((N+1)*(N+1)) + y*(N+1) + (z+1)];

    return vertices;
}

std::tuple<torch::Tensor, torch::Tensor, int, int> pred_to_mesh(torch::Tensor offset,
                 torch::Tensor topology,
                 torch::Tensor vertices_all,
                 torch::Tensor faces_all,
                 torch::Tensor vertice_number,
                 torch::Tensor face_number){

    int N = offset.size(1) - 1;
    int vertice_cnt = 0;
    int face_cnt = 0;

    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            for (int k=0; k<N; k++){
                int t = topology[i*N*N+j*N+k].item<int>();
                float offset_c[3]={(float)i, (float)j, (float)k};
            
                torch::Tensor *vertices = offset_to_vertices(offset, i, j, k);
                for (int tri_ind = 0; tri_ind<visTopology[1][t]; tri_ind++){
                    for (int vertex_ind = 0; vertex_ind<3; vertex_ind++){
                        int topology_ind = visTopology[0][t];
                        
                        for (int _i=0; _i<3; _i++){
                            vertices_all[(12*(32*32*32))*vertice_cnt + _i] = vertices[_i*3 + triTable[topology_ind][tri_ind*3+vertex_ind]] + offset_c[_i];
                            faces_all[(12*(32*32*32))*face_cnt + vertex_ind] = vertice_cnt;
                            vertice_cnt++;
                        }
                        face_cnt++;
                    }        
                }
            }
        }
    }

    vertice_number[0] = vertice_cnt;
    face_number[0] = face_cnt;

    return std::make_tuple(vertices_all, faces_all, vertice_cnt, face_cnt);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pred_to_mesh", &pred_to_mesh, "Convert predictions to mesh");
}