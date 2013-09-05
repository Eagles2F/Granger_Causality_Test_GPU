/*
	Description: This code tries to solve the inverse linear problem through 
	the traditional gradient descent method	on CUDA-enabled GPU
	Author:Yifan Li
	Org:Melady lab of USC
*/

//System includes
#include <stdio.h>
#include <math.h>
#include "mat.h"

//CUDA Runtime
#include <cuda_runtime.h>

// Matrices are stored in row-major order: 
// M(row, col) = *(M.elements + row * M.width + col) 
typedef struct {
    int width;    
  int height;    
	double* elements; 
	}Matrix;
	
//device function

//Vector * Matrix
__device__ void Vec_x_Matrix(double A[],Matrix B,double C[],int N,int M){
	int i,j;
	for(i=0; i < M; i++){
		double sum = 0.0;
		for(j=0; j < N; j++)
		{
			sum += A[j]*B.elements[j*B.width+i];
		}
		C[i] = sum;
	}
}

//Vector Substraction
__device__ void Vec_sub(double A[],double B[],double C[],int N){
	int i;
	for(i = 0; i < N; i++){
		C[i] = A[i]-B[i];
	}
}

//Constant x Vector
__device__ void Con_x_Vector(double Con, double * Vec, int N,double* Vec_R){
	int i;
	for(i=0;i<N;i++){
		Vec_R[i]=Vec[i]*Con;
	}
}

//host function

//Get sub_matrix from matrix
 void Sub_Matrix(Matrix m,Matrix Sub_m,int N,int M,int start,int end){
	int i,j;
	for(i=0; i< N; i++){
		for(j=start; j <end+1; j++){
			Sub_m.elements[i*Sub_m.width+j-start] = m.elements[i*m.width+j];
		}
	}
}


//Kernel Function
__global__  void Gradient_descent(Matrix A, Matrix Series,int max_iter, double delta,int N,int T,Matrix Series_sub1,Matrix Series_sub2,Matrix Series_sub1_Transpose){
	// create the vector for the intermediate result
	double* G=(double*)malloc(N*sizeof(double));
	double* Temp=(double*)malloc((T-1)*sizeof(double));
	double* Aind=(double*)malloc(N*sizeof(double));
	double* Series_sub2ind=(double*)malloc((T-1)*sizeof(double));

	int i,j;		
		
	// compute row ind of A in this thread 
	int ind = threadIdx.x; 

	for(i =0; i < N; i++){
		Aind[i]=A.elements[ind*A.width+i];
	}
	for(i=0; i < T-1; i++){
		Series_sub2ind[i]=Series_sub2.elements[ind*Series_sub2.width+i];
	}

	// Compute A[ind]
	for (i = 0; i < max_iter; i++){
		// compute G
		Vec_x_Matrix(Aind,Series_sub1,Temp,N,T-1);
		Vec_sub(Temp,Series_sub2ind,Temp,T-1);
		Vec_x_Matrix(Temp,Series_sub1_Transpose,G,T-1,N);
	
		// compute the A[ind] in the next iteration
		Con_x_Vector(delta,G,N,G);
		Vec_sub(Aind,G,Aind,N);
	}
	
	// Copy Aind to A[ind][:]
	for(i =0; i<N;i++){
		A.elements[ind*A.width+i]=Aind[i];
	}

	// Free the tempory memomry
	free(Series_sub2ind);
	free(Aind);
	free(Temp);
	free(G);	
}


int main(int argc, char** argv){
	double delta = (double)pow(10,-5);
	int max_iter = 100;
	size_t size;
	
	//load Series from .mat file
	MATFile *pmat;	
	const char* file =argv[1];
	const char* varname="Series";
	mxArray* Series_mat;
	pmat = matOpen(file, "r");
	if(pmat == NULL){
		printf("Error reopening file%s\n", file);
		return(NULL);
	}
	
	Series_mat =  matGetVariable(pmat, varname);
	if(Series_mat == NULL){
		printf("Error reading in file%s\n", file);
		return(NULL);
	}
	
	matClose(pmat);
	
	mwSize row, col; // mwSize is int 
	mwSize nRow = mxGetM(Series_mat); 
	mwSize nCol = mxGetN(Series_mat);
	double *Series_Pr = mxGetPr(Series_mat);
	int T = nCol;
	int N = nRow;	

	Matrix Series;
	Series.width = 	nCol;
	Series.height = nRow;
	size = Series.width*Series.height*sizeof(double);
	Series.elements = (double*)malloc(size);
	for(row = 0; row < nRow; row++) {
		for(col = 0; col < nCol; col++) {
			Series.elements[row*Series.width+col] = Series_Pr[nRow * col + row];// this needs testing
		}
	}
	
	//create matrix A
	Matrix A;
	A.width = N;
	A.height = N;
	size = A.width*A.height*sizeof(double);
	A.elements = (double*)malloc(size);
	int i,j;
	for(i =0; i < A.height; i++){
		for(j = 0; j < A.width; j++){
			A.elements[i*A.width+j] = 0;
		}
	}

        // allocate the space for Series_sub1
	Matrix Series_sub1;
	Series_sub1.width = T-1;
	Series_sub1.height = N;
	size = Series_sub1.width*Series_sub1.height*sizeof(double);
	Series_sub1.elements=(double*)malloc(size);

	// allocate the space for Series_sub2
	Matrix Series_sub2;
	Series_sub2.width = T-1;
	Series_sub2.height = N;
	size = Series_sub2.width*Series_sub2.height*sizeof(double);
	Series_sub2.elements=(double*)malloc(size);

	
	// allocate the space for Series_sub1_Transpose
	Matrix Series_sub1_Transpose;
	Series_sub1_Transpose.width = N;
	Series_sub1_Transpose.height = T-1;
	size = Series_sub1_Transpose.width*Series_sub1_Transpose.height*sizeof(double);
	Series_sub1_Transpose.elements=(double*)malloc(size);

	// compute sub matrix
	Sub_Matrix(Series,Series_sub1,N,T,0,T-2);
	Sub_Matrix(Series,Series_sub2,N,T,1,T-1);
	// transpose sub matrix
	for(i = 0;i<T-1;i++){
		for(j=0;j<N;j++){
			Series_sub1_Transpose.elements[i*Series_sub1_Transpose.width+j]=Series_sub1.elements[j*Series_sub1.height+i];
		}
	}
	
	//allocate device memory for Series_sub1	
	Matrix d_Series_sub1;
	d_Series_sub1.width = Series_sub1.width;
	d_Series_sub1.height = Series_sub1.height;
	size = Series_sub1.width*Series_sub1.height*sizeof(double);
	cudaMalloc(&d_Series_sub1.elements,size);
	cudaMemcpy(d_Series_sub1.elements,Series_sub1.elements,size,cudaMemcpyHostToDevice);

	//allocate device memory for Series_sub2	
	Matrix d_Series_sub2;
	d_Series_sub2.width = Series_sub2.width;
	d_Series_sub2.height = Series_sub2.height;
	size = Series_sub2.width*Series_sub2.height*sizeof(double);	
	cudaMalloc(&d_Series_sub2.elements,size);
	cudaMemcpy(d_Series_sub2.elements,Series_sub2.elements,size,cudaMemcpyHostToDevice);

	//allocate device memory for Series_sub1_Transpose
	Matrix d_Series_sub1_Transpose;
	d_Series_sub1_Transpose.width = Series_sub1_Transpose.width;
	d_Series_sub1_Transpose.height = Series_sub1_Transpose.height;
	size = Series_sub1_Transpose.width*Series_sub1_Transpose.height*sizeof(double);
	cudaMalloc(&d_Series_sub1_Transpose.elements,size);
	cudaMemcpy(d_Series_sub1_Transpose.elements,Series_sub1_Transpose.elements,size,cudaMemcpyHostToDevice);

	//allocate device memory for A
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size = d_A.width*d_A.height*sizeof(double);
	cudaMalloc(&d_A.elements,size);


	//allocate device memory for Series(To do: allocate memeory for Series in Surface memory)
	Matrix d_Series;
	d_Series.width = Series.width;
	d_Series.height = Series.height;
	size = d_Series.width*d_Series.height*sizeof(double);
	cudaMalloc(&d_Series.elements,size);
	cudaMemcpy(d_Series.elements, Series.elements, size, cudaMemcpyHostToDevice);

	// cuda timer	
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);
	int counter = 0;
	const int COUNTER=10;
	while(counter < COUNTER){
		//d_A initialization
		size=d_A.height*d_A.width*sizeof(double);
		cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
		//device code
		Gradient_descent<<<1,N>>>(d_A,d_Series,max_iter,delta,N,T,d_Series_sub1,d_Series_sub2,d_Series_sub1_Transpose);
		counter++;
	}
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	size = A.width*A.height*sizeof(double);
	cudaMemcpy(A.elements,d_A.elements,size,cudaMemcpyDeviceToHost);

	cudaFree(d_Series.elements);
	cudaFree(d_A.elements);
	cudaFree(d_Series_sub1_Transpose.elements);
	cudaFree(d_Series_sub2.elements);
	cudaFree(d_Series_sub1.elements);
	float ms;
	cudaEventElapsedTime(&ms,start,stop);
	printf("counter:%d  time required:%fms\n",COUNTER,ms);


	free(Series_sub1_Transpose.elements);
	free(Series_sub2.elements);
	free(Series_sub1.elements);
	free(A.elements);
	free(Series.elements);
	
	return 0;
}
