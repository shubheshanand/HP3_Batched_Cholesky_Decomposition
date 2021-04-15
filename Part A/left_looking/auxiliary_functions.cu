#include "./headers.h"


/*
 * Function to perform Choleshky Factorization for a tile
 * input is the pointer to shared memory for a tile given by t_A
 */
 __device__ void spotrf_tile(float* t_A)
 {

    int ty = threadIdx.x;  		// col
    int tx = threadIdx.y; 		// row
 
    for(int k{0};k<TILE_SIZE;k++){
    	
    	// square root of diagonal elements
        if(tx==0 && ty==0)
            t_A[k*TILE_SIZE + k] = sqrtf(t_A[k*TILE_SIZE + k]);
        __syncthreads();
 
        // division step done parallaly
        if(ty<tx && ty == k)
        {
            t_A[(tx)*TILE_SIZE + k]/= t_A[k*TILE_SIZE + k];
        }
        __syncthreads();
 
        if(ty<tx && ty > k)
        {
            t_A[(tx)*TILE_SIZE + (ty)]-= t_A[(tx)*TILE_SIZE + k]*t_A[(ty)*TILE_SIZE + k];
        }
        __syncthreads();
    }
 }

/*
 * Function to perform triangular solve for a tile
 * inputs are two shared memory pointer of tiles given by t_A1 and t_A2
 * implemnting triangular solve on tile t_A2 using t_A1 
 */

__device__ void strsm_tile(float *t_A1, float *t_A2)
{
    // t_A2 is current unkonown 
    
    int ty = threadIdx.x;        // access column
    int tx = threadIdx.y;       // access row
    
    for(int i{0};i<TILE_SIZE;i++){
        if(ty==0){
            t_A2[tx*TILE_SIZE + i]/= t_A1[i*TILE_SIZE + i];    // divison step
        }
        __syncthreads();

        if(ty>i && i<TILE_SIZE-1)
        {
            t_A2[tx*TILE_SIZE+ty]-= t_A2[tx*TILE_SIZE + i]*t_A1[ty*TILE_SIZE + i];
        }
        __syncthreads();
    }
 
}

/*
 * Function to perform rank-k update 
 * half of the threads working
 * inputs are pointers to the shared memory for two tiles given by rA1 and rA2
 * implementing rank-k update of the tile rA2 using tile rA1
 */

__device__ void ssyrk_tile(float* rA1, float* rA2) 
{
    
    int row = threadIdx.y;
    int column = threadIdx.x;

    if(column <= row)
    {
        float updatedValue = rA2[row * TILE_SIZE + column];

        for(int k=0; k<TILE_SIZE; k++)
        {
            updatedValue -= rA1[row * TILE_SIZE + k] * rA1[column * TILE_SIZE + k];
        }

        rA2[row * TILE_SIZE + column] = updatedValue;
    }
}

/*
 * Function to perform general matrix multiplication 
 * DOUBT: I think calculation is given wrong in paper it should be rA2[k][n] we are taking in row major form
 * inputs are pointers to the shared memory for three tiles given by rA1, rA2 and rA3
 * implementing sgemm on tile rA3 using rA1 and rA2
 */
__device__ void sgemm_tile(const float* rA1, const float* rA2, float* rA3)
{

    int row = threadIdx.y;
    int column = threadIdx.x;


    float updatedValue = rA3[row * TILE_SIZE + column];

    for(int i=0; i<TILE_SIZE; i++)
    {
        updatedValue -= rA1[row * TILE_SIZE + i] * rA2[i * TILE_SIZE + column];
    }

    rA3[row * TILE_SIZE + column] = updatedValue;
}

/*
 * Function to store full tile from shared memory back to global memory
 * inputs are pointers to tile of shared memory and global memory given by s_mem and g_mem
 * tile_y and tile_x are integers representing tile access numbers in y and x dimensions 
 */

 __device__ void store_full(float *g_mem, float *s_mem, int tile_y, int tile_x, int N)
{
    int tx = threadIdx.x;               // local threadid in x
    int ty = threadIdx.y;               // local threadid in y

    int row = tile_y * TILE_SIZE + ty;       // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        g_mem[row * N + column] = (tx < TILE_SIZE && ty < TILE_SIZE) ? s_mem[ty * TILE_SIZE + tx] : 0;
    }
    // __syncthreads();
}

/*
 * Function to store lower triangular tile from shared memory to global memory  
 * inputs are pointers to tile of shared memory and global memory given by s_mem and g_mem
 * tile_y and tile_x are integers representing tile access numbers in y and x dimensions and N is matrix size
 */
 __device__ void store_lower(float *g_mem, float *s_mem, int tile_y, int tile_x, int N)
 {
    int tx = threadIdx.x;               // local threadid in x
    int ty = threadIdx.y;               // local threadid in y

    int row = tile_y * TILE_SIZE + ty;      // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        g_mem[row * N + column] = (tx < TILE_SIZE && ty < TILE_SIZE && column <= row) ? s_mem[ty * TILE_SIZE + tx] : 0;
    }
    // __syncthreads();
 }

/*
 * Function to load a full tile from global memory to shared memory
 * inputs are pointers to tile of shared memory and global memory given by s_mem and g_mem
 * tile_y and tile_x are integers representing tile access numbers in y and x dimensions and N is matrix size
 */
__device__ void load_full(float *g_mem, float *s_mem, int tile_y, int tile_x, int N)
{
    int tx = threadIdx.x;                   // local threadid in x
    int ty = threadIdx.y;                   // local threadid in y

    int row = tile_y * TILE_SIZE + ty;      // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(tx < TILE_SIZE && ty < TILE_SIZE)
    {
        s_mem[ty * TILE_SIZE + tx] = (row < N && column < N) ? g_mem[row * N + column] : 0;
    }
    // __syncthreads();
}


/*
 * function to store 0 element in in global memory tile given by g_mem 
 * tile_y and tile_x are integers representing tile access numbers in y and x dimensions and N is matrix size
 */
 __device__ void store_zeros(float *g_mem, int tile_y, int tile_x, int N)
 {
    int tx = threadIdx.x;                   // local threadid in x
    int ty = threadIdx.y;                   // local threadid in y

    int row = tile_y * TILE_SIZE + ty;      // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        g_mem[row * N + column] = 0;
    }
    // __syncthreads();
 }
