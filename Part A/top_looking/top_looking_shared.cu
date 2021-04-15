#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>       

#define _TILE_SIZE 32
__constant__ int TILE_SIZE;

/*The algorithm and code given in the main reference paper have been followed*/
/*All matrices stored and accessed in row major form*/

/*Function to perform rank-k update */
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

/*General Matrix Multiplication*/
__device__ void sgemm_tile(float* rA1, float* rA2, float* rA3)
{
    int row = threadIdx.y;
    int column = threadIdx.x;    


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
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
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
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	for(int i=0;i<TILE_SIZE;i++)
  {
		if(tx==0)
    {
			t_A2[ty*TILE_SIZE + i] /= t_A1[i*TILE_SIZE + i];
		}
		__syncthreads();

		if(tx>i && tx<TILE_SIZE-1)
    {
			t_A2[ty*TILE_SIZE+tx] -= (t_A2[ty*TILE_SIZE + i]*t_A1[tx*TILE_SIZE + i]);
		}
		__syncthreads();
	}
 
}

__device__ void load_full_tile(int m, int n, float* g_in, float* arr, int N)
{
    int  i = m*TILE_SIZE + threadIdx.y;
    int  j = n*TILE_SIZE + threadIdx.x;
    if(!(i<N && j<N))  return;
     
    arr[threadIdx.y*TILE_SIZE + threadIdx.x] = g_in[i*N + j];
    __syncthreads(); 
}

__device__ void store_full_tile(int m, int n, float* g_in, float* arr, int N)
{
    int  i = m*TILE_SIZE + threadIdx.y;
    int  j = n*TILE_SIZE + threadIdx.x;
    if(!(i<N && j<N)) return;
    g_in[i*N + j] = arr[threadIdx.y*TILE_SIZE + threadIdx.x];
    __syncthreads(); 
}

__device__ void load_lower_tile(int m, int n, float* g_in, float* arr, int N)
{
    int  i = m*TILE_SIZE + threadIdx.y;
    int  j = n*TILE_SIZE + threadIdx.x;
    if(!(i<N && j<N)) return;
    
    if(threadIdx.x<=threadIdx.y)
     arr[threadIdx.y*TILE_SIZE + threadIdx.x] = g_in[i*N + j];
    else arr[threadIdx.y*TILE_SIZE + threadIdx.x] = 0.0;
    __syncthreads(); 
}

__device__ void store_lower_tile(int m, int n, float* g_in, float* arr, int N)
{
    int  i = m*TILE_SIZE + threadIdx.y;
    int  j = n*TILE_SIZE + threadIdx.x;
    if(!(i<N && j<N)) return;
    if(threadIdx.x<=threadIdx.y)
     g_in[i*N + j] = arr[threadIdx.y*TILE_SIZE + threadIdx.x];
    else g_in[i*N + j] = 0.0;
    __syncthreads(); 
}


void print_matrix(float *A,int m,int n, FILE* file_w)
{
    for(int i =0;i<m;i++)
    {
        for(int j=0;j<n;j++)
            fprintf(file_w, "%f ",A[i*n+j]);
        fprintf(file_w, "\n");
    }
}

__global__ void launch_kernel(float* d_mat, int N)
{
    
    extern __shared__ float shared_mem[];
    float *rA1, *rA2, *rA3;
    rA1 = shared_mem;  rA2 = shared_mem + TILE_SIZE*TILE_SIZE;   rA3 = shared_mem + 2*TILE_SIZE*TILE_SIZE; 
    
    int nn,kk,mm;
    int num_tiles = (N/TILE_SIZE) + ((N%TILE_SIZE)!=0); 
    for(kk=0; kk<num_tiles; kk++)
    {
      for(nn=0; nn<kk; nn++)
      {
        load_full_tile(kk,nn,d_mat,rA3,N);   
               
        for(mm=0; mm<nn; mm++)
        {
            load_full_tile(kk,mm,d_mat,rA1,N);
            load_full_tile(nn,mm,d_mat,rA2,N);
            sgemm_tile(rA1,rA2,rA3);
            __syncthreads(); 

        }
        
        load_lower_tile(nn,nn,d_mat,rA1,N);
        strsm_tile(rA1, rA3);
        __syncthreads();
        store_full_tile(kk,nn,d_mat,rA3,N);
      } 
      
      load_lower_tile(kk,kk,d_mat,rA1,N);
      
      for(nn=0; nn<kk; nn++)
      {
          load_full_tile(kk,nn,d_mat,rA2,N);
          ssyrk_tile(rA2,rA1);
          __syncthreads();
      }
      
      spotrf_tile(rA1);
      //__syncthreads();
      store_lower_tile(kk,kk,d_mat,rA1,N);
    }
 
    rA1[threadIdx.y*TILE_SIZE + threadIdx.x] = 0.0;
    __syncthreads();
 
    for(kk=0; kk<num_tiles; kk++)
     for(nn=kk+1; nn<num_tiles; nn++) 
       store_full_tile(kk,nn,d_mat,rA1,N);
       
}

int main(int argc,char *argv[])
{

    FILE *file_r = fopen(argv[1],"r");
    FILE *file_w = fopen(argv[2],"w");

    int N, TILE;
    // printf("Enter order of matrix: ");
    fscanf(file_r, "%d", &N);
    // printf("\nEnter the tile size: ");
    // scanf("%d", &TILE);
    TILE = _TILE_SIZE;
    
    size_t size = N*N*sizeof(float);
    float* h_mat = (float*) malloc(size);
    // init_mat(h_mat);
    printf("\nReading input matrix: ");
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j++)
        {
            fscanf(file_r, " %f",&h_mat[i*N + j]);
        }
    }

    fclose(file_r);
    //print_matrix(h_mat, N, N);
    float* d_mat;     
    cudaMalloc((void **)&d_mat, size); 
    cudaMemcpy(d_mat, h_mat, size, cudaMemcpyHostToDevice);   
    
    cudaMemcpyToSymbol(TILE_SIZE, &TILE, sizeof(TILE));    
    dim3 block(TILE,TILE,1);
    printf("\nPerforming top looking cholesky factorization...\n\n");
    launch_kernel<<<1,block,3*TILE*TILE*sizeof(float)>>> (d_mat, N);
    cudaMemcpy(h_mat, d_mat, size, cudaMemcpyDeviceToHost);
    print_matrix(h_mat, N, N, file_w);

    fclose(file_w);
    return  0;
}
