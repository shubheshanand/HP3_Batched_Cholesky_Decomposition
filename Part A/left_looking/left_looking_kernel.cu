#include "./headers.h"
#include "./auxiliary_functions.cu"

/*
 * Left looking kernel code (without loop unrolling)
 * INPUT: g_in: pointer of matrix present in Global memory of GPU
 *        N: Matrix size
 * 
 * Stores the lower triangular matrix back in g_in
 */
 __global__ void left_looking_kernel(float *g_in, int N)
 {
 
     // (ceil(N / TILE_SIZE) + 2) * sizeof(TILE) amount of shared memory
     // sizeof(TILE) = TILE_SIZE * (TILE_SIZE + 1)  // TILE_SIZE + 1 to resolve bank conflicts in strsm, sgemm and ssyrk functions
     extern __shared__ float s_current_panel[];
 
     // Pointers for accessing shared memory locations
     float *rA1 = NULL;
     float *rA2 = NULL;
     float *rA3 = NULL;
 
     // no of tiles in a column
     int no_of_tiles = (N / TILE_SIZE) + (N % TILE_SIZE != 0); // ceil (N / TILE_SIZE)
 
 
     // i: current panel
     for(int i=0; i<no_of_tiles; i++)
     {
 
         // loading current panel in shared memory
         for(int j=i; j<no_of_tiles; j++)
         {
             rA1 = s_current_panel + j * TILE_SIZE * (TILE_SIZE + 1);
             load_full(g_in, rA1, j, i, N);
         }
         __syncthreads();
 
 
         // UPDATE CURRENT PANEL using preceding panels
         // j: preceding panel no.
         for(int j=0; j<i; j++)
         {
             // Loading data for rank-k update in shared memory
             rA1 = s_current_panel + no_of_tiles * TILE_SIZE * (TILE_SIZE + 1);
             load_full(g_in, rA1, i, j, N);
             __syncthreads();
 
 
             // Rank-k update
             rA1 = s_current_panel + no_of_tiles * TILE_SIZE * (TILE_SIZE + 1);
             rA2 = s_current_panel + i * TILE_SIZE * (TILE_SIZE + 1);
             
             ssyrk_tile(rA1, rA2);
             __syncthreads();
 
 
             // Applying SGEMM 
             for(int k=i+1; k<no_of_tiles; k++)
             {
                 // Loading data for sgemm in shared memory
                 rA1 = s_current_panel + (no_of_tiles + 1) * TILE_SIZE * (TILE_SIZE + 1);
                 load_full(g_in, rA1, k, j, N);
                 __syncthreads();
 
 
                 // sgemm
                 rA1 = s_current_panel + no_of_tiles * TILE_SIZE * (TILE_SIZE + 1);
                 rA2 = s_current_panel + (no_of_tiles + 1) * TILE_SIZE * (TILE_SIZE + 1);
                 rA3 = s_current_panel + k * TILE_SIZE * (TILE_SIZE + 1);
 
                 sgemm_tile(rA1, rA2, rA3);
                 __syncthreads();
             }
 
         }
 
 
         // FACTORIZE CURRENT PANEL
         
         // applying spotrf on the tile (i, i)
         rA1 = s_current_panel + i * TILE_SIZE * (TILE_SIZE + 1);
 
         spotrf_tile(rA1);
         __syncthreads();
 
         
         // Applying TRSM
         for(int k=i+1; k<no_of_tiles; k++)
         {
             // trsm
             rA2 = s_current_panel + k * TILE_SIZE * (TILE_SIZE + 1);
 
             strsm_tile(rA1, rA2);
             __syncthreads();
         }
 
 
 
         // STORING the current panel back in the global memory
         for (int k=0; k<no_of_tiles; k++)
         {
             rA1 = s_current_panel + k * TILE_SIZE * (TILE_SIZE + 1);
 
             // store zeros for tiles above the tile (i, i)
             if(k < i)
             {
                 store_zeros(g_in, k, i, N);
             }
             else
             {
                 // store lower for tile (i, i)
                 if(k == i)
                 {
                     store_lower(g_in, rA1, k, i, N);
                 }
                 else // store full for tiles below the tile (i, i)
                 {
                     store_full(g_in, rA1, k, i, N);
                 }
             }
         }
         
 
         __syncthreads();
     }
 }


 /*
 * Left looking kernel code with the use of less shered memory (without loop unrolling)
 * INPUT: g_in: pointer of matrix present in Global memory of GPU
 *        N: Matrix size
 * 
 * Stores the lower triangular matrix back in g_in
 */
 __global__ void left_looking_kernel_less_mem(float *g_in, int N)
 {
     extern __shared__ float s_current_panel[];
 
 
     // Pointers for accessing shared memory locations
     float *rA1 = NULL;
     float *rA2 = NULL;
     float *rA3 = NULL;
 
     // no of tiles in a column
     int no_of_tiles = (N / TILE_SIZE) + (N % TILE_SIZE != 0);    // ceil(N / TILE_SIZE)
 
 
     // i: current panel
     for(int i=0; i<no_of_tiles; i++)
     {

         // loading current panel in shared memory
         for(int j=0; j<no_of_tiles; j++)
         {
             rA1 = s_current_panel + j * TILE_SIZE * (TILE_SIZE + 1);
             load_full(g_in, rA1, j, i, N);
         }
         __syncthreads();
 
 
         for(int j=0; j<no_of_tiles; j++)
         {
 
             if(j >= i)
             {
                 if(j == i)         // representing the tile on which spotrf will be carried out
                 {
                     for(int k=0; k<i; k++)         // k iterates over tiles left of (i,i) tile
                     {
 
                         //rA2 = &s_current_panel[2 * TILE_SIZE * TILE_SIZE]; 
                         rA2 = s_current_panel + 2 * TILE_SIZE * (TILE_SIZE + 1);    
                         load_full(g_in, rA2, j, k, N);
                         __syncthreads();
 
                         //rA1 = &s_current_panel[0];
                         rA1 = s_current_panel + 0 * TILE_SIZE * (TILE_SIZE + 1);
                         ssyrk_tile(rA1, rA2);                                  // rank-k update on rA1 using rA2
                         __syncthreads();
 
                     }
 
 
                     //rA1 = &s_current_panel[0];
                     rA1 = s_current_panel + 0 * TILE_SIZE * (TILE_SIZE + 1);
                     spotrf_tile(rA1);
                     __syncthreads();
 
                     store_lower(g_in, rA1, i, i, N);                   // storing (i,i) tile back to global memory after calling sporf 
                     __syncthreads();
                }
                 else
                 {
 
                     //rA3 = &s_current_panel[1 * TILE_SIZE * TILE_SIZE];
                      rA3 = s_current_panel + 1 * TILE_SIZE * (TILE_SIZE + 1);
                     load_full(g_in, rA3, j, i, N);
                     __syncthreads();
 
                     for(int k=0; k<i; k++)                             // k iterates over tile below (i,i) tile
                     {
 
                         //rA1 = &s_current_panel[2 * TILE_SIZE * TILE_SIZE];
                         rA1 = s_current_panel + 2 * TILE_SIZE * (TILE_SIZE + 1);
                         load_full(g_in, rA1, i, k, N);
 
                         //rA2 = &s_current_panel[3 * TILE_SIZE * TILE_SIZE];
                         rA2 = s_current_panel + 3 * TILE_SIZE * (TILE_SIZE + 1);
                         load_full(g_in, rA2, j, k, N);
                         __syncthreads();
 
 
                         sgemm_tile(rA1, rA2, rA3);                     // sgemm on tile rA3 using tiles rA1 and rA2
                         __syncthreads();
 
                     }
 
 
                     //rA1 = &s_current_panel[0];
                     rA1 = s_current_panel + 0 * TILE_SIZE * (TILE_SIZE + 1);
                     //rA2 = &s_current_panel[1 * TILE_SIZE * TILE_SIZE];
                     rA2 = s_current_panel + 1 * TILE_SIZE * (TILE_SIZE + 1);
 
                     strsm_tile(rA1, rA2);                              // strsm on tile rA2 using tile rA1
                     __syncthreads();
 
                     store_full(g_in, rA2, j, i, N);                    // storing back to global memory
                     __syncthreads();
                 }
 
             }
             else
             {
                 store_zeros(g_in, j, i, N);                            // stores zero in the tile given by pointer g_in
                 __syncthreads();
             }
             
         }
         
         __syncthreads();
     }
 }