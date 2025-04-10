#define TILE_SIZE 16

__kernel void matmul(__global const float* A, 
                     __global const float* B, 
                     __global float* C, 
                     const unsigned int N) 
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if(row >= N || col >= N) return;
    
    __local float A_tile[TILE_SIZE][TILE_SIZE];
    __local float B_tile[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;
    
    for(int t = 0; t < (N + TILE_SIZE - 1)/TILE_SIZE; t++) {
        // Load A tile (TILE_SIZE x TILE_SIZE block from A's row)
        int a_col = t*TILE_SIZE + get_local_id(1);
        A_tile[get_local_id(0)][get_local_id(1)] = 
            (a_col < N) ? A[row*N + a_col] : 0.0f;
        
        // Load B tile (TILE_SIZE x TILE_SIZE block from B's column) with transposed storage
        int b_row = t*TILE_SIZE + get_local_id(0);
        B_tile[get_local_id(1)][get_local_id(0)] =  // Transposed storage
            (b_row < N) ? B[b_row*N + col] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum with proper tile alignment
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[get_local_id(0)][k] * B_tile[get_local_id(1)][k];  // Note k index swap
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    C[row*N + col] = sum;
}
