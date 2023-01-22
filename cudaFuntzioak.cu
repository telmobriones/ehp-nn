#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <typeinfo>
#include "../include/cudaFuntzioak.cuh"


#define THR_PER_BLOCK 1024 


__global__ void cuda_input_layer(nn_t *nn, double *input, double **A, double **Z){
  
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i<nn->layers_size[0]){
       A[0][i]=input[i];
   }
}

__global__ void cuda_matrix_mul_add(double *Z, double *W, double *A, int W_rows, int W_cols, int A_rows, int A_cols, double *B){

    // W1-en dimentsioak (60x30)=1800 elementu > THR_PER_BLOCK ?!?!?!
    __shared__ float tmp[THR_PER_BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(  i < W_rows * A_cols){
        tmp[threadIdx.x] = W[i] * A[i % A_rows];
        //tmp[threadIdx.x] += B[i % A_rows];
        __syncthreads();
        
        if(threadIdx.x == 0){
            float batura=0.0;
                for(int j=0; j<THR_PER_BLOCK;j++){
                        batura += tmp[j];
                }
            Z[i % A_rows] = batura;
 	    }
    }
/*   
    //Bi dimentsioekin saiatu gara baina Segmentation fault
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int Z_cols = A_cols;
    int Z_rows = W_rows;
    float Z_val = 0;
    if (row < Z_rows && col < Z_cols) {
        for (int i = 0; i < W_cols; i++) {
            Z_val += W[row * W_cols + i] * A[i * A_cols + col];
        }
        Z[row * Z_cols + col] = Z_val + B[row];
    }*/

}

__global__ void cuda_matrix_func(double *n, double *m, int rows, int cols){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(  i< rows * cols){
        n[i]=1 / (1+ exp(-m[i]));
    }	
}



double cuda_forward_pass(nn_t *nn, double *input, double **A, double **Z){
  
    cudaEvent_t start, stop;
    double **d_A, **d_Z;
    float milliseconds = 0;
    int batura = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /*
    gpuErrchk(cudaMalloc((void **)&d_A, nn->n_layers * sizeof(double*)));
    gpuErrchk(cudaMalloc((void **)&d_Z, nn->n_layers * sizeof(double*)));

    for(int i = 0; i < nn->n_layers; i++) {
        gpuErrchk(cudaMalloc((void **)&d_A[i], nn->layers_size[i] * sizeof(double)));
        gpuErrchk(cudaMalloc((void **)&d_Z[i], nn->layers_size[i] * sizeof(double)));
        // Copy A and Z to device memory
        gpuErrchk(cudaMemcpy(d_A[i], A[i], nn->layers_size[i]*sizeof(double), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Z[i], Z[i], nn->layers_size[i]*sizeof(double), cudaMemcpyHostToDevice));
    }
    */

    printf("total number of rows: %ld\n",sizeof(A)/sizeof(A[0]));
    printf("total number of columns: %ld\n",sizeof(A[0])/sizeof(char));

    //   gpuErrchk(cudaMalloc((void **)&d_A, nn->n_layers * sizeof(double)));
    //   gpuErrchk(cudaMalloc((void **)&d_Z, nn->n_layers * sizeof(double)));   

    for(int i=0; i<nn->n_layers;i++){
        batura += nn->layers_size[i];
    //       gpuErrchk(cudaMalloc(&d_A[i], nn->layers_size[i] * sizeof(double)));
    //       gpuErrchk(cudaMalloc(&d_Z[i], nn->layers_size[i] * sizeof(double)));
    }

    gpuErrchk(cudaMalloc(&d_A, nn->n_layers * batura * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_Z, nn->n_layers * batura * sizeof(double)));

    // Copy data from host arrays A and B to device arrays d_A and d_Z
    gpuErrchk(cudaMemcpy(d_A, A, nn->n_layers * batura * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Z, Z, nn->n_layers * batura * sizeof(double), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil( (float)nn->layers_size[0] / thr_per_blk );
    
    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_input_layer<<<blk_in_grid, thr_per_blk>>>(nn, input, d_A, d_Z);
    gpuErrchk(cudaEventRecord(stop));


    
    //dim3 block_size(256);
    //dim3 num_of_blocks((nn->n_layers * batura + block_size.x - 1) / block_size.x);
    

    printf("batura : %d\n",batura*nn->n_layers);

    for (int i=0; i<nn->n_layers; i++){

        blk_in_grid = ceil( (float)nn->layers_size[i]/ thr_per_blk );
        //dim3 block_size(256);
        //dim3 num_of_blocks((nn->n_layers * batura + block_size.x - 1) / block_size.x);
        gpuErrchk(cudaEventRecord(start));
        cuda_matrix_mul_add<<<blk_in_grid, thr_per_blk>>>(d_Z[i], nn->WH[i - 1], d_A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);
        //cuda_matrix_mul_add<<<num_of_blocks, block_size>>>(d_Z[i], nn->WH[i - 1], d_A[i - 1],  nn->layers_size[i], nn->layers_size[i - 1], nn->layers_size[i - 1], 1, nn->BH[i - 1]);
        gpuErrchk(cudaEventRecord(stop));
        printf("batura : %d\n",batura*nn->n_layers);
        gpuErrchk(cudaEventRecord(start));
        cuda_matrix_func<<<blk_in_grid, thr_per_blk>>>(A[i], Z[i], nn->layers_size[i], 1);
        gpuErrchk(cudaEventRecord(stop));
        printf("aaaaa\n");

    }

    // Copy data from device array to host array
    cudaMemcpy(A, d_A, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Z, d_Z, sizeof(double), cudaMemcpyDeviceToHost);


    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_Z);


    return(milliseconds);
}
