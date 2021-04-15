#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include <math.h>       // needed for the function sqrtf()

#define TILE_SIZE 32 // NB // Block SIZE

// char input_file[] = "InputFiles/num_1024_dim_30.txt";
// char output_file[] = "./output.txt";

//AUX FUNCTIONS

/*
 * Function to perform Choleshky Factorization for a tile
 * input is the pointer to shared memory for a tile given by t_A
 */
 __device__ void spotrf_tile(float* t_A)
 {

    int ty = threadIdx.y;  		// col
    int tx = threadIdx.z; 		// row
 
    for(int k{0};k<TILE_SIZE;k++){
    	
    	// square root of diagonal elements
        if(tx==0 && ty==0)
            t_A[k*TILE_SIZE + k] = sqrtf(t_A[k*TILE_SIZE + k]);
        __syncthreads();
 
        // division step done parallaly
        if(ty<=tx && tx<TILE_SIZE - 1 && ty<TILE_SIZE - 1 && ty == k)
        {
            t_A[(tx+1)*TILE_SIZE + k]/= t_A[k*TILE_SIZE + k];
        }
        __syncthreads();
 
        if(ty<=tx && tx<TILE_SIZE - 1 && ty<TILE_SIZE - 1 && ty >= k)
        {
            t_A[(tx+1)*TILE_SIZE + (ty+1)]-= t_A[(tx+1)*TILE_SIZE + k]*t_A[(ty+1)*TILE_SIZE + k];
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
    
    int ty = threadIdx.y;        // access column
    int tx = threadIdx.z;       // access row
    
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

/*
 * Function to perform general matrix multiplication 
 * DOUBT: I think calculation is given wrong in paper it should be rA2[k][n] we are taking in row major form
 * inputs are pointers to the shared memory for three tiles given by rA1, rA2 and rA3
 * implementing sgemm on tile rA3 using rA1 and rA2
 */
__device__ void sgemm_tile(const float* rA1, const float* rA2, float* rA3)
{

    int row = threadIdx.z;
    int column = threadIdx.y;


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

 __device__ void store_full(float *g_mem, float *s_mem, int tile_y, int tile_x, int N, int M, int shared_size_single_matrix)
{
    int tx = threadIdx.y;               // local threadid in x
    int ty = threadIdx.z;               // local threadid in y

    int row = tile_y * TILE_SIZE + ty;       // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        g_mem[blockIdx.x * blockDim.x + threadIdx.x + row * N * M + column*M] = (tx < TILE_SIZE && ty < TILE_SIZE) ? s_mem[ty * TILE_SIZE + tx + shared_size_single_matrix*threadIdx.x] : 0;
    }
    __syncthreads();
}

/*
 * Function to store lower triangular tile from shared memory to global memory  
 * inputs are pointers to tile of shared memory and global memory given by s_mem and g_mem
 * tile_y and tile_x are integers representing tile access numbers in y and x dimensions and N is matrix size
 */
 __device__ void store_lower(float *g_mem, float *s_mem, int tile_y, int tile_x, int N, int M, int shared_size_single_matrix)
 {
    int tx = threadIdx.y;               // local threadid in x
    int ty = threadIdx.z;               // local threadid in y

    int row = tile_y * TILE_SIZE + ty;      // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        g_mem[blockIdx.x * blockDim.x + threadIdx.x + row * N * M + column*M] = (tx < TILE_SIZE && ty < TILE_SIZE && column <= row) ? s_mem[ty * TILE_SIZE + tx + shared_size_single_matrix*threadIdx.x] : 0;
    }
    __syncthreads();
 }

/*
 * Function to load a full tile from global memory to shared memory
 * inputs are pointers to tile of shared memory and global memory given by s_mem and g_mem
 * tile_y and tile_x are integers representing tile access numbers in y and x dimensions and N is matrix size
 */
 __device__ void load_full(float *g_mem, float *s_mem, int tile_y, int tile_x, int N, int M, int shared_size_single_matrix)
 {
    int tx = threadIdx.x;                   // local threadid in x
    int ty = threadIdx.y;                   // local threadid in y
    int tz = threadIdx.z; 
    //printf("%d %d %d \n",tx,ty,tz);     
    int row = tile_y * TILE_SIZE + tz;      // access row
    int column = tile_x * TILE_SIZE + ty;   // access col
    if(ty < TILE_SIZE && tz < TILE_SIZE && tx<M)
    {
        s_mem[tz * TILE_SIZE + ty + shared_size_single_matrix*tx] = (row < N && column < N) ? g_mem[ blockIdx.x * blockDim.x + tx + row * N * M + column*M] : 0; // we need to think about access expression of global memory. //M: Total number of matrices. N:dim of matrix
    }
    __syncthreads();
}


/*
 * function to store 0 element in in global memory tile given by g_mem 
 * tile_y and tile_x are integers representing tile access numbers in y and x dimensions and N is matrix size
 */
 __device__ void store_zeros(float *g_mem, int tile_y, int tile_x, int N, int M)
 {
    int tx = threadIdx.y;                   // local threadid in x
    int ty = threadIdx.z;                   // local threadid in y

    int row = tile_y * TILE_SIZE + ty;      // access row
    int column = tile_x * TILE_SIZE + tx;   // access col

    if(row < N && column < N)
    {
        g_mem[blockIdx.x * blockDim.x +  threadIdx.x + row * N * M + column*M] = 0;
    }
    __syncthreads();
 }


/* LEFT LOOKING KERNEL FUNCTIONS */

  __global__ void left_looking_kernel(float *g_in, int N, int M , int shared_size_single_matrix)
 {
 
     // (ceil(N / TILE_SIZE) + 2) * sizeof(TILE) amount of shared memory
     extern __shared__ float s_current_panel[];
 
     // Pointers for accessing shared memory locations
     float *rA1 = NULL;
     float *rA2 = NULL;
     float *rA3 = NULL;

     int tx = threadIdx.x;
     // no of tiles in a column
     int no_of_tiles = (N / TILE_SIZE) + (N % TILE_SIZE != 0); // ceil (N / TILE_SIZE)
 
     // i: current panel
     for(int i=0; i<no_of_tiles; i++)
     {
 
         // loading current panel in shared memory
         for(int j=0; j<no_of_tiles; j++)
         {
             rA1 = &s_current_panel[j * TILE_SIZE * TILE_SIZE];
             load_full(g_in, rA1, j, i, N, M, shared_size_single_matrix);
         }
         __syncthreads();
 
 
         // UPDATE CURRENT PANEL using preceding panels
         // j: preceding panel no.
         for(int j=0; j<i; j++)
         {
             // Loading data for rank-k update in shared memory
             rA1 = &s_current_panel[no_of_tiles * TILE_SIZE * TILE_SIZE];
             load_full(g_in, rA1, i, j, N, M, shared_size_single_matrix);
             __syncthreads();
 
 
             // Rank-k update
             rA1 = &s_current_panel[tx*shared_size_single_matrix +no_of_tiles * TILE_SIZE * TILE_SIZE];
             rA2 = &s_current_panel[tx*shared_size_single_matrix +i * TILE_SIZE * TILE_SIZE];
             
             ssyrk_tile(rA1, rA2);
             __syncthreads();
 
 
             // Applying SGEMM 
             for(int k=i+1; k<no_of_tiles; k++)
             {
                 // Loading data for sgemm in shared memory
                 rA1 = &s_current_panel[(no_of_tiles + 1) * TILE_SIZE * TILE_SIZE];
                 load_full(g_in, rA1, k, j, N , M, shared_size_single_matrix);
                 __syncthreads();
 
 
                 // sgemm
                 rA1 = &s_current_panel[tx*shared_size_single_matrix +no_of_tiles * TILE_SIZE * TILE_SIZE];
                 rA2 = &s_current_panel[tx*shared_size_single_matrix +(no_of_tiles + 1) * TILE_SIZE * TILE_SIZE];
                 rA3 = &s_current_panel[tx*shared_size_single_matrix +k * TILE_SIZE * TILE_SIZE];
 
                 sgemm_tile(rA1, rA2, rA3);
                 __syncthreads();
             }
 
         }
 
 
         // FACTORIZE CURRENT PANEL
         
         // applying spotrf on the tile (i, i)
         rA1 = &s_current_panel[tx*shared_size_single_matrix +i * TILE_SIZE * TILE_SIZE];
 
         spotrf_tile(rA1);
         __syncthreads();
 
         
         // Applying TRSM
         for(int k=i+1; k<no_of_tiles; k++)
         {
             // trsm
             rA2 = &s_current_panel[tx*shared_size_single_matrix +k * TILE_SIZE * TILE_SIZE];
 
             strsm_tile(rA1, rA2);
             __syncthreads();
         }
 
 
 
         // STORING the current panel back in the global memory
         for (int k=0; k<no_of_tiles; k++)
         {
             rA1 = &s_current_panel[k * TILE_SIZE * TILE_SIZE];
 
             // store zeros for tiles above the tile (i, i)
             if(k < i)
             {
                 store_zeros(g_in, k, i, N, M);
             }
             else
             {
                 // store lower for tile (i, i)
                 if(k == i)
                 {
                     store_lower(g_in, rA1, k, i, N, M, shared_size_single_matrix);
                 }
                 else // store full for tiles below the tile (i, i)
                 {
                     store_full(g_in, rA1, k, i, N, M, shared_size_single_matrix);
                 }
             }
         }
         
 
         __syncthreads();
     }
 }


__global__ void left_looking_kernel_less_mem(float *g_in, int N, int M , int shared_size_single_matrix)
 {
     extern __shared__ float s_current_panel[];
 
 
     // Pointers for accessing shared memory locations
     float *rA1 = NULL;
     float *rA2 = NULL;
     float *rA3 = NULL;
 
     // no of tiles in a column
     int no_of_tiles = (N / TILE_SIZE) + (N % TILE_SIZE != 0);    // ceil(N / TILE_SIZE)
     int tx = threadIdx.x;
 
     // i: current panel
     for(int i=0; i<no_of_tiles; i++)
     {
 
         // loading tile(i, i)
         rA1 = &s_current_panel[0];
         load_full(g_in, rA1, i, i, N, M, shared_size_single_matrix);
 
 
         for(int j=0; j<no_of_tiles; j++)
         {
 
             if(j >= i)
             {
                 if(j == i)         // representing the tile on which spotrf will be carried out
                 {
                     for(int k=0; k<i; k++)         // k iterates over tiles left of (i,i) tile
                     {
 
                         rA2 = &s_current_panel[2 * TILE_SIZE * TILE_SIZE];     
                         load_full(g_in, rA2, j, k, N , M, shared_size_single_matrix);
                         rA2 = &s_current_panel[tx*shared_size_single_matrix + 2 * TILE_SIZE * TILE_SIZE]; 
                         rA1 = &s_current_panel[tx*shared_size_single_matrix + 0];
                         ssyrk_tile(rA1, rA2);                                  // rank-k update on rA1 using rA2
                         __syncthreads();
 
                     }
 
 
                     rA1 = &s_current_panel[tx*shared_size_single_matrix + 0];
                     spotrf_tile(rA1);
                     __syncthreads();
                     rA1 = &s_current_panel[0];
                     store_lower(g_in, rA1, i, i, N, M, shared_size_single_matrix);                   // storing (i,i) tile back to global memory after calling sporf 
                 }
                 else
                 {
 
                     rA3 = &s_current_panel[1 * TILE_SIZE * TILE_SIZE];
                     load_full(g_in, rA3, j, i, N, M, shared_size_single_matrix);
 
                     for(int k=0; k<i; k++)                             // k iterates over tile below (i,i) tile
                     {
 
                         rA1 = &s_current_panel[2 * TILE_SIZE * TILE_SIZE];
                         load_full(g_in, rA1, i, k, N, M, shared_size_single_matrix);
 
                         rA2 = &s_current_panel[tx*shared_size_single_matrix + 3 * TILE_SIZE * TILE_SIZE];
                         load_full(g_in, rA1, j, k, N, M, shared_size_single_matrix);
 
                         rA1 = &s_current_panel[tx*shared_size_single_matrix + 2 * TILE_SIZE * TILE_SIZE];
                         rA3 = &s_current_panel[tx*shared_size_single_matrix + 1 * TILE_SIZE * TILE_SIZE];
                         sgemm_tile(rA1, rA2, rA3);                     // sgemm on tile rA3 using tiles rA1 and rA2
                         __syncthreads();
 
                     }
 
 
                     rA1 = &s_current_panel[tx*shared_size_single_matrix + 0];
                     rA2 = &s_current_panel[tx*shared_size_single_matrix + 1 * TILE_SIZE * TILE_SIZE];
 
                     strsm_tile(rA1, rA2);                              // strsm on tile rA2 using tile rA1
                     __syncthreads();
                     rA2 = &s_current_panel[1 * TILE_SIZE * TILE_SIZE];
                     store_full(g_in, rA2, j, i, N, M, shared_size_single_matrix);                    // storing back to global memory
                 }
 
             }
             else
             {
                 store_zeros(g_in, j, i, N, M);                            // stores zero in the tile given by pointer g_in
             }
             
         }
         
         __syncthreads();
     }
 }



//MAIN PROGRAM


 int main(int argc,char *argv[]) {


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

    for (int matrix_index = 0; matrix_index < num_of_matrices; matrix_index++)
    {
        for (int row = 0; row < dim_of_matrix; row++)
        {
            for (int column = 0; column < dim_of_matrix; column++)
            {
                fscanf(fptr, "%f", &read_element);
                global_id = row * dim_of_matrix * num_of_matrices + column * num_of_matrices + matrix_index;
                h_A[global_id] = read_element;
            }
        }
    }
    printf("\nRead from the input file successfully!\n");
    fclose(fptr);

    printf("\nPrinting the host-side input array read from the input file:\n");
    for (int i = 0; i < numElements; i++) {    
        printf("%f ", h_A[i]);
    }
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


    int num_of_matrices_per_block = min(1024/(TILE_SIZE * TILE_SIZE) , num_of_matrices);
    dim3 grid(num_of_matrices / num_of_matrices_per_block , 1, 1);
    dim3 block(num_of_matrices_per_block, TILE_SIZE, TILE_SIZE);
    // no of tiles in a column
    int INPUT_SIZE = dim_of_matrix;
    int no_of_tiles = (INPUT_SIZE / TILE_SIZE) + (INPUT_SIZE % TILE_SIZE != 0); // ceil of (INPUT_SIZE / TILE_SIZE)
    if(TILE_SIZE == INPUT_SIZE)
    {
        left_looking_kernel<<<grid, block, num_of_matrices_per_block * 1 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,1 * TILE_SIZE * TILE_SIZE);
    }

    else if((no_of_tiles + 2) * TILE_SIZE * TILE_SIZE * sizeof(float) < devp.sharedMemPerBlock)
    {
        left_looking_kernel<<<grid, block,num_of_matrices * (no_of_tiles + 2) * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,(no_of_tiles + 2) * TILE_SIZE * TILE_SIZE);
    }
    else
    {
        left_looking_kernel_less_mem<<<grid, block, num_of_matrices_per_block * 4 * TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_A, dim_of_matrix, num_of_matrices ,4 * TILE_SIZE * TILE_SIZE);
    }


    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        printf("kernel launch failed with error \"%s\".\n",
                cudaGetErrorString(cudaerr));
    }


    // COPY BACK TO HOST, FREE CUDA MEM, HOST MEM, AND RESET CUDA


    err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    else {
        printf("\nCopied d_A to host side successfully!\n");
    }
    
    printf("\nPrinting the output of cudememcopyDeviceToHost, i.e. the host-side array returned from device side:\n");
    for (int i = 0; i < numElements; i++) {    
        printf("%f ", h_A[i]);
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


    FILE *fptr1;
    fptr1 = fopen(argv[2], "w+");
    float write_element;
    fprintf(fptr1, "%d\n", num_of_matrices);
    fprintf(fptr1, "%d\n", dim_of_matrix);

    for (int matrix_index = 0; matrix_index < num_of_matrices; matrix_index++)
    {
        for (int row = 0; row < dim_of_matrix; row++)
        {
            for (int column = 0; column < dim_of_matrix; column++)
            {
                global_id = row * dim_of_matrix * num_of_matrices + column * num_of_matrices + matrix_index;
                write_element = h_A[global_id] ;
                fprintf(fptr1, "%0.2f ", write_element);
            }
         fprintf(fptr1,"\n");
        }
        fprintf(fptr1,"\n");
    }
    fclose(fptr1);
    free(h_A);
    printf("\n\nAll tasks completed successfully!\n\n");
    return 0;
}