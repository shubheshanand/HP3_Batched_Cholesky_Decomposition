#include "./headers.h"
#include "./left_looking_kernel.cu"

int main(int argc,char *argv[])
{

    FILE *file_r = fopen(argv[1],"r");
    FILE *file_w = fopen(argv[2],"w");

    cudaError_t err = cudaSuccess;

    int devCount;
    cudaGetDeviceCount(&devCount);

    cudaDeviceProp devp;
    cudaGetDeviceProperties(&devp, 0);

    int INPUT_SIZE = 0;
    fscanf(file_r, "%d", &INPUT_SIZE);

    size_t size = INPUT_SIZE * INPUT_SIZE * (sizeof(float));
    printf("Testing for matrix M [%dx%d]\n", INPUT_SIZE, INPUT_SIZE);

    float *M = (float *)malloc(size);

    if(M == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    printf("Reading input matrix: \n");
    for(int i=0; i<INPUT_SIZE; i++)
    {
        for(int j=0; j<INPUT_SIZE; j++)
        {
            fscanf(file_r, "%f ", &M[i * INPUT_SIZE + j]);
        }
    }

    // printf("Printing input matrix\n");
    // for(int i=0; i<INPUT_SIZE; i++)
    // {
    //     for(int j=0; j<INPUT_SIZE; j++)
    //     {
    //         printf("%f ", M[i * INPUT_SIZE + j]);
    //     }
    //     printf("\n");
    // }

    printf("\n\n");
    fclose(file_r);

    float *d_M = NULL;
    err = cudaMalloc((void **)&d_M, size);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate matrix M on the CUDA device! (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    printf("Copy the matrix M from host memory to CUDA device\n\n");

    err = cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix M from host to device (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 grid(1, 1, 1);
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    // no of tiles in a column
    int no_of_tiles = (INPUT_SIZE / TILE_SIZE) + (INPUT_SIZE % TILE_SIZE != 0); // ceil of (INPUT_SIZE / TILE_SIZE)

    if(TILE_SIZE == INPUT_SIZE)
    {
        left_looking_kernel<<<grid, block, 1 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float)>>>(d_M, INPUT_SIZE);
    }
    else if((no_of_tiles + 2) * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float) < devp.sharedMemPerBlock)
    {
        left_looking_kernel<<<grid, block, (no_of_tiles + 2) * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float)>>>(d_M, INPUT_SIZE);
    }
    else
    {
        left_looking_kernel_less_mem<<<grid, block, 4 * TILE_SIZE * (TILE_SIZE + 1) * sizeof(float)>>>(d_M, INPUT_SIZE);
    }
    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch CUDA kernel (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(M, d_M, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Printing output matrix\n");
    for(int i=0; i<INPUT_SIZE; i++)
    {
        for(int j=0; j<INPUT_SIZE; j++)
        {
            fprintf(file_w, "%f ", M[i * INPUT_SIZE + j]);
        }
        fprintf(file_w, "\n");
    }

    err = cudaFree(d_M);
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(M);

    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the CUDA device (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    fclose(file_w);

    printf("DONE!\n");

}