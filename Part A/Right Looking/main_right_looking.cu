#include "right_looking.cu"
int main(int argc,char *argv[])
{
	FILE *fil_r = fopen(argv[1],"r");
    FILE *fil_w = fopen(argv[2],"w");
    int n,N;
    //printf("Enter dimension (N) : ");
    fscanf(fil_r,"%d",&n);
    if((n%TILE_SIZE)==0)
        N = n;
    else
        N = (((int) (n/TILE_SIZE)) + 1)*TILE_SIZE;
    size_t size = N*N*sizeof(float);
    float *M = (float *)malloc(size);
    if(M == NULL)
    {
        fprintf(fil_w,"Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    int i,j;
    //printf("Enter input matrix: \n");
    for(i=0;i<N;i++)
    {
        for(j=0;j<N;j++)
        {
            if(i>=n || j>=n)
                M[i*N + j] = 1;     //Padding the matrix with 1
            else
                fscanf(fil_r,"%f",&M[i*N + j]);
        }
    }
    fclose(fil_r);
    cudaError_t err = cudaSuccess;
    float *read_data = NULL;
    err = cudaMalloc((void **)&read_data,N*N*sizeof(float));
    if(err != cudaSuccess)
    {
        fprintf(fil_w,"Failed to allocate matrix on the CUDA device! (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    fprintf(fil_w,"Coping the matrix from host memory to device memory\n");
    err = cudaMemcpy(read_data,M,size,cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        fprintf(fil_w,"Failed to copy matrix from host to device (error code %s)\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    fprintf(fil_w,"Testing for matrix M [%dx%d]\n",N,N);
    dim3 grid(1,1,1);
    dim3 block(TILE_SIZE,TILE_SIZE,1);
    size_t shared_size = (N*TILE_SIZE + TILE_SIZE*TILE_SIZE + 1)*sizeof(float);
    right_looking_launch_kernel<<<grid,block,shared_size>>>(read_data,N);
    err = cudaMemcpy(M,read_data,size,cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
        fprintf(fil_w,"Failed to copy the output matrix M from device to Host (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    fprintf(fil_w,"Printing output matrix\n");
    for(i=0;i<n;i++)
    {
        for(j=0;j<n;j++)
            fprintf(fil_w,"%f\t",M[i*N + j]);
        fprintf(fil_w,"\n");
    }
    err = cudaFree(read_data);
    if(err != cudaSuccess)
    {
        fprintf(fil_w,"Failed to free device matrix M (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceReset();
    if(err != cudaSuccess)
    {
        fprintf(fil_w,"Failed to deinitialize the CUDA device (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    free(M);
    fprintf(fil_w,"DONE!\n");
    fclose(fil_w);
    return 0;
}