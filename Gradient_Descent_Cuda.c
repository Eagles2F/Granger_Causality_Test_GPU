#include <stdio.h>
#include <stdlib.h>
#include "mat.h"
#include "matrix.h"

// Matrices are stored in row-major order: 
// M(row, col) = *(M.elements + row * M.width + col) 
typedef struct {
    int width;    
  int height;    
	double* elements; 
	}Matrix;
	
//Vector * Matrix
__device__ void Vec_x_Matrix(double A[],Matrix B,double C[],int N,int M){
	for(int i=0; i < M; i++){
		double sum = 0.0;
		for(int j=0; j < N; j++)
		{
			sum += A[j]*B.elements[j*B.width+i];
		}
		C[i] = sum;
	}
}

//Vector Substraction
__device__ void Vec_sub(double A[],double B[],double C[],int N){
	for(int i = 0; i < N; i++){
		C[i] = A[i]-B[i];
	}
}

//Constant x Vector
__device__ void Con_x_Vector(double Con, double * Vec, int N,double* Vec_R){
	for(int i=0;i<N;i++){
		Vec_R[i]=Vec[i]*Con;
	}
}

//Get sub_matrix from matrix
__device__ void Sub_Matrix(Matrix m;Matrix Sub_m,int N,,int M,int start,int end){
	for(int i=0; i< N; i++){
		for(int j=start; j <end+1; j++){
			Sub_m.elements[i*Sub_m.width+j-start] = m.elements[i*Sub_m.width+j];
		}
	}
}


//Kernel Function
__global__  void Gradient_descent(Matrix A, Matrix Series,int max_iter, double delta,int N,int T){
	// create the vector for the intermediate result
	double* G = malloc(N*sizeof(double)); 
	double* Temp = malloc((T-1)*sizeof(double));
	double* Aind = (double*)malloc(N*sizeof(double));
	double* Series_sub2ind = (double*)malloc((T-1)*sizeof(double));

	// allocate the space for Series_sub1
	Matrix Series_sub1;
	Series_sub1.width = T-1;
	Series_sub1.height = N;
	size = Series_sub1.width*Series_sub1.height*sizeof(double);
	Series_sub1.elements = malloc(size);

	// allocate the space for Series_sub2
	Matrix Series_sub2;
	Series_sub2.width = T-1;
	Series_sub2.height = N;
	size = Series_sub2.width*Series_sub2.height*sizeof(double);
	Series_sub2.elements = malloc(size);

	
	// allocate the space for Series_sub1_Transpose
	Matrix Series_sub1_Transpose;
	Series_sub1_Transpose.width = N;
	Series_sub1_Transpose.height = T-1;
	size = Series_sub1_Transpose.width*Series_sub1_Transpose.height*sizeof(double);
	Series_sub1_Transpose.elements = malloc(size);
	
		
	// compute row ind of A in this thread 
	int ind = threadIdx.x; 

	// compute sub matrix
	Sub_Matrix(Series,Series_sub1,N,T,0,T-2);
	Sub_Matrix(Series,Series_sub2,N,T,1,T-1);
	// transpose sub matrix1
	for(int i = 0;i<T-1;i++){
		for(int j=0;j<N;j++){
			Series_sub1_Transpose.elements[i*Series_sub1_Transpose.width+j]=Series_sub1.elements[j*Series_sub1.height+i];
		}
	}

	for(int i =0; i < N; i++){
		Aind[i]=A.elements[ind*A.width+i];
	}
	for(int i=0; i < T-1; i++){
		Series_sub2ind[i]=Series_sub2.elements[ind*Series_sub2.width+i];
	}

	// Compute A[ind]
	for (int i = 0; i < max_iter; i++){
		// compute G
		Vec_x_Matrix(Aind,Series_sub1,Temp,N,T-1);
		Vec_sub(Temp,Series_sub2ind,Temp,T-1);
		Vec_x_Matrix(Temp,Series_sub1_Transpose,G,T-1,N);
	
		// compute the A[ind] in the next iteration
		Con_x_Vector(delta,G,N,G);
		Vec_sub(Aind,G,Aind,N);
	}
	
	// Free the tempory memomry
	free(Series_sub1_Transpose.elements);
	free(Series_sub2.elements);
	free(Series_sub1.elements);
	free(Series_sub2ind);
	free(Aind);
	free(Temp);
	free(G);
	
}


int main(int argc, char** argv){
	int T = 200;
	int N = 100;
	double delta = (double)exp(-5);
	int max_iter = 100;
	size_t size;
	
	//load Series from .mat file
	MATFile *pmat;	
	const char* file ="Series.mat";
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
	
	matClose(matFILE);
	
	mwSize row, col; // mwSize is int 
	mwSize nRow = mxGetM(Series_mat); 
	mwSize nCol = mxGetN(Series_mat);
	double *Series_Pr = mxGetPr(Series_mat);
	
	Matrix Series;
	Series.width = 	nCol;
	Series.height = nRow;
	size = Series.width*Series.height*sizeof(double);
	Series.elements = malloc(size);
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
	A.elements = malloc(size);
	for(int i =0; i < A.height; i++){
		for(int j = 0; j < A.width; j++){
			A.elements[i*A.width+j] = 0;
		}
	}
	
	//allocate device memory for A
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size = d_A.width*d_A.height*sizeof(double);
	cudaMalloc(&d_A.elements,size);
	cudaMemcopy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
	
	//allocate device memory for Series(To do: allocate memeory for Series in Surface memory)
	Matrix d_Series;
	d_Series.width = Series.width;
	d_Series.height = Series.height;
	size = d_Series.width*d_Series.height*sizeof(double);
	cudaMalloc(&d_Series.elements,size);
	cudaMemcopy(d_Series.elements, Series.elements, size, cudaMemcpyHostToDevice);
	
	//copy the parameters into device memory
	__device__ int d_N = N;
	__device__ int d_T = T;
	__device__ int d_max_iter = max_iter;
	__device__ double d_delta = delta;
		
	//device code
	Gradient_descent<<<1,N>>>(d_A,d_Series,d_max_iter,d_delta,d_N,d_T);
	
	cudaFree(d_Series);
	cudaFree(d_A);
	free(A);
	free(Series);
	return 0;
}
