#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>       

#define TILE_SIZE 8
#define chunk_size 32

// char input_file[] = "InputFiles/num_1024_dim_50.txt";
// char output_file[] = "./output.txt";


/*The algorithm and code given in the main reference paper have been followed*/
/*All matrices stored and accessed in row major form*/


/*Function to perform rank-k update */
__device__ void ssyrk_tile(float* rA1, float* rA2) 
{
    int row = threadIdx.z;
    int column = threadIdx.y;

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

/*General Matrix Multiplication*/
__device__ void sgemm_tile(float* rA1, float* rA2, float* rA3)
{
    int row = threadIdx.z;
    int column = threadIdx.y;    


    float updatedValue = rA3[row * TILE_SIZE + column];

    for(int i=0; i<TILE_SIZE; i++)
    {
        updatedValue -= rA1[row * TILE_SIZE + i] * rA2[column*TILE_SIZE + i];
    }

    rA3[row * TILE_SIZE + column] = updatedValue;
}

/*Function to perform Cholesky Factorization for a tile*/
 __device__ void spotrf_tile(float* t_A)
{
    int t_x = threadIdx.y;
    int t_y = threadIdx.z;
    for(int k=0;k<TILE_SIZE;k++)
    {
        if(t_x==t_y && t_x==k)
            t_A[k*TILE_SIZE + k] = sqrtf(t_A[k*TILE_SIZE + k]);
        __syncthreads();
        
        if(t_x<t_y && t_x == k && t_x<TILE_SIZE && t_y<TILE_SIZE)
        {
            t_A[t_y*TILE_SIZE + k]/= t_A[k*TILE_SIZE + k];
        }
        __syncthreads();
        
        if(t_x<=t_y && t_x>k && t_y>k && t_x<TILE_SIZE && t_y<TILE_SIZE)
        {
            t_A[t_y*TILE_SIZE + t_x]-= t_A[t_x*TILE_SIZE + k]*t_A[t_y*TILE_SIZE + k];
        }
        __syncthreads();
    }
}


/*Function to perform triangular solve for a tile */

__device__ void strsm_tile(float *t_A1, float *t_A2)
{
	int tx = threadIdx.y;
	int ty = threadIdx.z;
	
	for(int i=0;i<TILE_SIZE;i++)
  {
		if(tx==0)
    {
			t_A2[ty*TILE_SIZE + i] /= t_A1[i*TILE_SIZE + i];
		}
		__syncthreads();

		if(tx>i && i<TILE_SIZE-1)
    {
			t_A2[ty*TILE_SIZE+tx] -= (t_A2[ty*TILE_SIZE + i]*t_A1[tx*TILE_SIZE + i]);
		}
		__syncthreads();
	}
 
}

__device__ void load_full_tile(float *g_mem, float *s_mem, int tile_y, int tile_x, int N, int M, int shared_size_single_matrix)
 {
    int tx = threadIdx.x;                   // local threadid in x
    int ty = threadIdx.y;                   // local threadid in y
    int tz = threadIdx.z; 
    //printf("%d %d %d \n",tx,ty,tz);     
    int row = tile_y * TILE_SIZE + tz;      // access row
    int column = tile_x * TILE_SIZE + ty;   // access col
    if(ty < TILE_SIZE && tz < TILE_SIZE && tx<M)
    {
        int g_threadX = blockIdx.x * blockDim.x +  threadIdx.x;
        int x = row*N + column;
        int global_id = ((g_threadX / chunk_size) *chunk_size)*N*N + x *chunk_size + (g_threadX % chunk_size);
       
        s_mem[tz * TILE_SIZE + ty + shared_size_single_matrix*tx] = (row < N && column < N) ? g_mem[ global_id] : 0; // we need to think about access expression of global memory. //M: Total number of matrices. N:dim of matrix
    }
    __syncthreads();
}

__device__ void store_full_tile(float *g_mem, float *s_mem, int tile_y, int tile_x, int N, int M, int shared_size_single_matrix)
{
    int tx = threadIdx.y;               // local threadid in x
    int ty = threadIdx.z;               // local threadid in y

    int row = tile_y * TILE_SIZE + ty;       // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        int g_threadX = blockIdx.x * blockDim.x +  threadIdx.x;
        int x = row*N + column;
        int global_id = ((g_threadX / chunk_size) *chunk_size)*N*N + x *chunk_size + (g_threadX % chunk_size);
       
        g_mem[global_id] = (tx < TILE_SIZE && ty < TILE_SIZE) ? s_mem[ty * TILE_SIZE + tx + shared_size_single_matrix*threadIdx.x] : 0;
    }
    __syncthreads();
}

__device__ void load_lower_tile(float *g_mem, float *s_mem, int tile_y, int tile_x, int N, int M, int shared_size_single_matrix)
{

    int tx = threadIdx.x;                   // local threadid in x
    int ty = threadIdx.y;                   // local threadid in y
    int tz = threadIdx.z; 
    int row = tile_y * TILE_SIZE + tz;      // access row
    int column = tile_x * TILE_SIZE + ty;   // access col
    if(ty < TILE_SIZE && tz < TILE_SIZE && tx<M)
    {
        int g_threadX = blockIdx.x * blockDim.x +  threadIdx.x;
        int x = row*N + column;
        int global_id = ((g_threadX / chunk_size) *chunk_size)*N*N + x *chunk_size + (g_threadX % chunk_size);
       
        if(threadIdx.y<=threadIdx.z)
            s_mem[tz * TILE_SIZE + ty + shared_size_single_matrix*tx] = (row < N && column < N) ? g_mem[ global_id] : 0; // we need to think about access expression of global memory. //M: Total number of matrices. N:dim of matrix
        else
            s_mem[tz * TILE_SIZE + ty + shared_size_single_matrix*tx] = 0.0;
    }
    __syncthreads();
}



__device__ void store_lower_tile(float *g_mem, float *s_mem, int tile_y, int tile_x, int N, int M, int shared_size_single_matrix)
 {
    int tx = threadIdx.y;               // local threadid in x
    int ty = threadIdx.z;               // local threadid in y

    int row = tile_y * TILE_SIZE + ty;      // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        int g_threadX = blockIdx.x * blockDim.x +  threadIdx.x;
        int x = row*N + column;
        int global_id = ((g_threadX / chunk_size) *chunk_size)*N*N + x *chunk_size + (g_threadX % chunk_size);
       
        g_mem[global_id] = (tx < TILE_SIZE && ty < TILE_SIZE && column <= row) ? s_mem[ty * TILE_SIZE + tx + shared_size_single_matrix*threadIdx.x] : 0;
    }
    __syncthreads();
 }
 __device__ void store_zeros(float *g_mem, int tile_y, int tile_x, int N, int M)
 {
    int tx = threadIdx.y;                   // local threadid in x
    int ty = threadIdx.z;                   // local threadid in y

    int row = tile_y * TILE_SIZE + ty;      // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        int g_threadX = blockIdx.x * blockDim.x +  threadIdx.x;
        int x = row*N + column;
        int global_id = ((g_threadX / chunk_size) *chunk_size)*N*N + x *chunk_size + (g_threadX % chunk_size);
       
        g_mem[global_id] = 0;
    }
    __syncthreads();
 }

 __device__ void print_matrix(float* h_A,int num_of_matrices,int dim_of_matrix)
{
    
    for (int matrix_index = 0; matrix_index < num_of_matrices; matrix_index++)
    {
        for (int row = 0; row < dim_of_matrix; row++)
        {
            for (int column = 0; column < dim_of_matrix; column++)
            {
                int global_id = row * dim_of_matrix * num_of_matrices + column * num_of_matrices + matrix_index;
                float write_element = h_A[global_id] ;
                printf("%0.2f ", write_element);
            }
         printf("\n");
        }
        printf("\n");
    }
}

__global__ void launch_kernel(float* d_mat, int N, int M , int shared_size_single_matrix)
{
    
    extern __shared__ float shared_mem[];
    float *rA1, *rA2, *rA3;
    rA1 = shared_mem;  rA2 = shared_mem + TILE_SIZE*TILE_SIZE;   rA3 = shared_mem + 2*TILE_SIZE*TILE_SIZE; 
    int tx = threadIdx.x;
    int nn,kk,mm;
    int num_tiles = (N/TILE_SIZE) + ((N%TILE_SIZE)!=0); 
    for(kk=0; kk<num_tiles; kk++)
    {
      for(nn=0; nn<kk; nn++)
      {
        load_full_tile(d_mat , rA3, kk,nn,N,M, shared_size_single_matrix);   
               
        for(mm=0; mm<nn; mm++)
        {
            load_full_tile(d_mat , rA1, kk,mm,N,M, shared_size_single_matrix);
            load_full_tile(d_mat , rA2, nn,mm,N,M, shared_size_single_matrix);
            sgemm_tile(&rA1[tx*shared_size_single_matrix],&rA2[tx*shared_size_single_matrix],&rA3[tx*shared_size_single_matrix]);
            __syncthreads(); 

        }
        
        load_lower_tile(d_mat , rA1, nn,nn,N,M, shared_size_single_matrix);

        strsm_tile(&rA1[tx*shared_size_single_matrix], &rA3[tx*shared_size_single_matrix]);
        __syncthreads();
        store_full_tile(d_mat , rA3, kk,nn,N,M, shared_size_single_matrix);
        
      } 
      
      load_lower_tile(d_mat , rA1, kk,kk,N,M, shared_size_single_matrix);

      for(nn=0; nn<kk; nn++)
      {
          load_full_tile(d_mat , rA2, kk,nn,N,M, shared_size_single_matrix);
          ssyrk_tile(&rA2[tx*shared_size_single_matrix],&rA1[tx*shared_size_single_matrix]);
          __syncthreads();
      }
      
      spotrf_tile(&rA1[tx*shared_size_single_matrix]);
      store_lower_tile(d_mat , rA1, kk,kk,N,M, shared_size_single_matrix);

    }
 
    rA1[threadIdx.z*TILE_SIZE + threadIdx.y] = 0.0;
    __syncthreads();
 
    for(kk=0; kk<num_tiles; kk++)
        for(nn=kk+1; nn<num_tiles; nn++)
        {
            if(kk < nn)
            {
                store_zeros(d_mat, kk, nn, N, M);
            }
            else
            {
                store_full_tile(d_mat , rA1, kk,nn,N,M, shared_size_single_matrix);
            }
        }
            
       
}

int main(int argc,char *argv[])
{
    // READ FROM THE INPUT FILE


    FILE *fptr;
    fptr = fopen(argv[1], "r");
    int num_of_matrices, dim_of_matrix;
    fscanf(fptr, "%d", &num_of_matrices);
    fscanf(fptr, "%d", &dim_of_matrix);
    float read_element;
    float* h_A = NULL;
    int numElements = num_of_matrices * dim_of_matrix * dim_of_matrix;
    size_t size = numElements * sizeof(float);
    cudaDeviceProp devp;
    cudaGetDeviceProperties(&devp, 0);

    h_A = (float *)malloc(size);
    
    int global_id = 0;
    for(int p = 0; p < (num_of_matrices/chunk_size); p++)
    {
        for (int matrix_index = 0; matrix_index < chunk_size; matrix_index++)
        {
            for (int row = 0; row < dim_of_matrix; row++)
            {
                for (int column = 0; column < dim_of_matrix; column++)
                {
                    fscanf(fptr, "%f", &read_element);
                    int x = row*dim_of_matrix + column;
                    global_id = (p*chunk_size)*dim_of_matrix*dim_of_matrix + x*chunk_size + matrix_index;
                    h_A[global_id] = read_element;
                }
            }
        }
    }

    printf("\nRead from the input file successfully!\n");
    fclose(fptr);

    printf("\n\n");


    // COPY TO DEVICE


    cudaError_t err = cudaSuccess;

    float *d_A = NULL;

    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else {
        printf("Copied the h_A to device side successfully!\n\n");
    }


    // LAUNCH KERNEL


    int num_of_matrices_per_block = min(min(1024/(TILE_SIZE * TILE_SIZE) , num_of_matrices), chunk_size);
    dim3 grid(num_of_matrices / num_of_matrices_per_block , 1, 1);
    dim3 block(num_of_matrices_per_block, TILE_SIZE, TILE_SIZE);
    // no of tiles in a column
    int INPUT_SIZE = dim_of_matrix;
    int no_of_tiles = (INPUT_SIZE / TILE_SIZE) + (INPUT_SIZE % TILE_SIZE != 0); // ceil of (INPUT_SIZE / TILE_SIZE)
    launch_kernel<<<grid, block, num_of_matrices_per_block * 3 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,3 * TILE_SIZE * TILE_SIZE);
    
    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else {
        printf("\nCopied d_A to host side successfully!\n");
    }
    
    err = cudaFree(d_A);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "\nFailed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the CUDA device (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
    }


    FILE *fptr1;
    fptr1 = fopen(argv[2], "w+");
    float write_element;
    fprintf(fptr1, "%d\n", num_of_matrices);
    fprintf(fptr1, "%d\n", dim_of_matrix);

    for(int p = 0; p < (num_of_matrices/chunk_size); p++)
    {
        for (int matrix_index = 0; matrix_index < chunk_size; matrix_index++)
        {
            for (int row = 0; row < dim_of_matrix; row++)
            {
                for (int column = 0; column < dim_of_matrix; column++)
                {
                    int x = row*dim_of_matrix + column;
                    global_id = (p*chunk_size)*dim_of_matrix*dim_of_matrix + x*chunk_size + matrix_index;
                    write_element = h_A[global_id];
                    fprintf(fptr1, "%0.2f ", write_element);
                }
                fprintf(fptr1, "\n");
            }
        }
    }
    fclose(fptr1);
    free(h_A);
    printf("\n\nAll tasks completed successfully!\n\n");
    return 0;
}
