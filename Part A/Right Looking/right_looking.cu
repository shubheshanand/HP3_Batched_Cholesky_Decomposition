#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#define TILE_SIZE 32            // Tile size and block size, both are taken as 32
__device__ void store_full_row(float*,float*,int,int);
__device__ void load_full_row(float*,float*,int,int);
__device__ void store_full(float*,float*,int,int,int);
__device__ void load_full(float*,float*,int,int,int);
__device__ void store_lower(float*,float*,int,int,int);
__device__ void load_lower(float*,float*,int,int,int);
__device__ void potrf_tile(float*);
__device__ void trsm_tile(float*,int,int,int);
__device__ void syrk_tile(float*,float*,int,int,int);
__device__ void store_zeros(float*,int);
__global__ void right_looking_launch_kernel(float*,int);

__device__ void store_full_row(float* read_data,float* write_data,int i,int N)
{
    int global_y;
    int global_x = i*blockDim.x + threadIdx.x;
    for(int j=0;j<N/TILE_SIZE;j++)
    {
        global_y = j*blockDim.y + threadIdx.y;
        write_data[global_y*N + global_x] = read_data[threadIdx.x + TILE_SIZE*global_y];
    }
    __syncthreads();
}
__device__ void load_full_row(float* read_data,float* write_data,int i,int N)
{
    int global_y;
    int global_x = i*blockDim.x + threadIdx.x;
    for(int j=0;j<N/TILE_SIZE;j++)
    {
        global_y = j*blockDim.y + threadIdx.y;
        write_data[threadIdx.x + TILE_SIZE*global_y] = read_data[global_y*N + global_x];
    }
    __syncthreads();
}
__device__ void store_full(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    write_data[global_y*N + global_x] = read_data[threadIdx.x + TILE_SIZE*threadIdx.y];
    __syncthreads();
}
__device__ void load_full(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = read_data[global_y*N + global_x];
    __syncthreads();
}
__device__ void store_lower(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    if(threadIdx.y >= threadIdx.x)
        write_data[global_y*N + global_x] = read_data[threadIdx.x + TILE_SIZE*threadIdx.y];
    else
        write_data[global_y*N + global_x] = 0.0;
    __syncthreads();
}
__device__ void load_lower(float* read_data,float* write_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    if(threadIdx.y >= threadIdx.x)
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = read_data[global_y*N + global_x];
    else
        write_data[threadIdx.x + TILE_SIZE*threadIdx.y] = 0.0;
    __syncthreads();
}
__device__ void potrf_tile(float* t_A)
{
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    __shared__ float temp2;
    for(int k=0;k<TILE_SIZE;k++)
    {
        if(t_x==t_y && t_x==k)
        {
            t_A[k*TILE_SIZE + k] = sqrtf(t_A[k*TILE_SIZE + k]);
            temp2 = t_A[k*TILE_SIZE + k];
        }
        __syncthreads();
        if(t_x<t_y && t_x == k)
        {
            t_A[t_y*TILE_SIZE + k]/= temp2;
        }
        __syncthreads();
        if(k<t_y && k<t_x && t_x<=t_y)
        {
            t_A[t_y*TILE_SIZE + t_x]-= t_A[t_x*TILE_SIZE + k]*t_A[t_y*TILE_SIZE + k];
        }
        __syncthreads();
    }
}
__device__ void trsm_tile(float *row_data,int i,int j,int N)
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;
    for(int s=0;s<TILE_SIZE;s++)
    {
	if(t_x==s)
    {
	    row_data[global_y*TILE_SIZE + t_x]/= row_data[global_x*TILE_SIZE + t_x];
	}
	__syncthreads();
	if(t_x > s)
    {
	    row_data[global_y*TILE_SIZE + t_x]-= row_data[global_x*TILE_SIZE +  s]*row_data[global_y*TILE_SIZE + s];
	}
	__syncthreads();
    }
}
__device__ void syrk_tile(float* row_data,float* edit_data,int i,int j,int N) 
{
    int global_y = j*blockDim.y + threadIdx.y;
    int global_x = i*blockDim.x + threadIdx.x;
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    float valueToSubtract = 0.0;
    for(int r=0;r<TILE_SIZE;r++)
    {
        valueToSubtract+= row_data[r + global_y*TILE_SIZE]*row_data[r + global_x*TILE_SIZE];
    }
    edit_data[t_y*TILE_SIZE + t_x]-= valueToSubtract;
    __syncthreads();
}
__device__ void store_zeros(float* A,int N)
{
    int t_y = threadIdx.y;
    int t_x = threadIdx.x;
    int i,j;
    for(i=0;i<N/TILE_SIZE-1;i++)
    {
        for(j=i+1;j<N/TILE_SIZE;j++)
            A[j*blockDim.x + t_x + (i*blockDim.y + t_y)*N] = 0.0;
    }
    __syncthreads();
}
__global__ void right_looking_launch_kernel(float* read_data,int N)
{
    __shared__ float tile_data[TILE_SIZE*TILE_SIZE];
    extern __shared__ float row_data[];
    int i,j,k;
    for(i=0;i<N/TILE_SIZE;i++)
    {
        load_lower(read_data,tile_data,i,i,N);
        potrf_tile(tile_data);
        store_lower(tile_data,read_data,i,i,N);
        load_full_row(read_data,row_data,i,N);
        for(j=i+1;j<N/TILE_SIZE;j++)
        {
            trsm_tile(row_data,i,j,N);
            for(k=i+1;k<j;k++)
            {
                load_full(read_data,tile_data,k,j,N);
                syrk_tile(row_data,tile_data,k,j,N);
                store_full(tile_data,read_data,k,j,N);
            }
            load_full(read_data,tile_data,k,j,N);
            syrk_tile(row_data,tile_data,k,j,N);
            store_full(tile_data,read_data,k,j,N);
        }
        store_full_row(row_data,read_data,i,N);
    }
    store_zeros(read_data,N);
}