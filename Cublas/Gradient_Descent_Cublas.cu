#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mat.h"

#include <cublas_v2.h>

#include <cuda_runtime.h>
// Matrices are stored in column-major order: 
// M(row, col) = M.elements[row + col*M.height]
typedef struct {
    	int width;    
 	int height;    
	double* elements; 
	}Matrix;

//Matrix * Matrix
//C(N,M)=A(N,N)*B(N,M)
void Matrix_x_Matrix(cublasHandle_t &handle,Matrix A,Matrix B,Matrix C){
	int N=A.width;
	int M=B.width;
	int lda=N,ldb=N,ldc=N;
	const double alf=1;
	const double bet=0;
	const double*alpha = &alf;
	const double*beta=&bet;
		
	//Do the operation
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,M,N,alpha,A.elements,lda,B.elements,ldb,beta,C.elements,ldc);
}

//Matrix - beta*Matrix
void Matrix_minus_Matrix(cublasHandle_t &handle,Matrix A,Matrix B, Matrix C,double beta){
	beta = -beta;
	double alpha = 1;
	const double* alf = &alpha;
	const double* bet = &beta;
	int lda = A.height;
	int ldb = B.height;
	int ldc = C.height;
	cublasDgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,A.height,A.width,alf,A.elements,lda,bet,B.elements,ldb,C.elements,ldc);
}

//Get sub_matrix from matrix
void Sub_Matrix(Matrix m,Matrix Sub_m,int N,int M,int start,int end){
	int i,j;
	for(i=0; i< N; i++){
		for(j=start; j <end+1; j++){
			Sub_m.elements[i+(j-start)*Sub_m.height] = m.elements[i+j*m.height];
		}
	}
}

void Gradient_descent(Matrix A, Matrix Series,int max_iter, double delta,int N,int T){
	
	// allocate the space for Series_sub1
	Matrix Series_sub1;
	Series_sub1.width = T-1;
	Series_sub1.height = N;
	size_t size = Series_sub1.width*Series_sub1.height*sizeof(double);
	Series_sub1.elements = (double*)malloc(size);

	// allocate the space for Series_sub2
	Matrix Series_sub2;
	Series_sub2.width = T-1;
	Series_sub2.height = N;
	size = Series_sub2.width*Series_sub2.height*sizeof(double);
	Series_sub2.elements = (double*)malloc(size);

	
	// allocate the space for Series_sub1_Transpose
	Matrix Series_sub1_Transpose;
	Series_sub1_Transpose.width = N;
	Series_sub1_Transpose.height = T-1;
	size = Series_sub1_Transpose.width*Series_sub1_Transpose.height*sizeof(double);
	Series_sub1_Transpose.elements = (double*)malloc(size);
	
	// compute sub matrix
	Sub_Matrix(Series,Series_sub1,N,T,0,T-2);
	Sub_Matrix(Series,Series_sub2,N,T,1,T-1);
	// transpose sub matrix1
	int i,j,ind;
	for(i = 0;i<T-1;i++){
		for(j=0;j<N;j++){
			Series_sub1_Transpose.elements[i+j*Series_sub1_Transpose.height]=Series_sub1.elements[j+i*Series_sub1.height];
		}
	}


	//allocate the device memory for Series_sub1,Series_sub1_Transpose,Temp,A,Series_sub2
	Matrix d_Series_sub1;
	d_Series_sub1.width = T-1;
	d_Series_sub1.height = N;
	size = d_Series_sub1.width*d_Series_sub1.height*sizeof(double);
	cudaMalloc(&d_Series_sub1.elements,size);
	cudaMemcpy(d_Series_sub1.elements,Series_sub1.elements,size,cudaMemcpyHostToDevice);

	Matrix d_Series_sub2;
	d_Series_sub2.width = T-1;
	d_Series_sub2.height = N;
	size = d_Series_sub2.width*d_Series_sub2.height*sizeof(double);
	cudaMalloc(&d_Series_sub2.elements,size);
	cudaMemcpy(d_Series_sub2.elements,Series_sub2.elements,size,cudaMemcpyHostToDevice);

	Matrix d_Series_sub1_Transpose;
	d_Series_sub1_Transpose.width = N;
	d_Series_sub1_Transpose.height = T-1;
	size = d_Series_sub1_Transpose.width*d_Series_sub1_Transpose.height*sizeof(double);
	cudaMalloc(&d_Series_sub1_Transpose.elements,size);
	cudaMemcpy(d_Series_sub1_Transpose.elements,Series_sub1_Transpose.elements,size,cudaMemcpyHostToDevice);

	Matrix d_A;
	d_A.width =A.width;
	d_A.height = A.height;
	size = d_A.width*d_A.height*sizeof(double);
	cudaMalloc(&d_A.elements,size);
	cudaMemcpy(d_A.elements,A.elements,size,cudaMemcpyHostToDevice);

	Matrix d_Temp;
	d_Temp.width = T-1;
	d_Temp.height = N;
	size = d_Temp.width*d_Temp.height*sizeof(double);
	cudaMalloc(&d_Temp.elements,size);

	Matrix d_G;
	d_G.width = N;
	d_G.height= N;
	size = d_G.width*d_G.height*sizeof(double);
	cudaMalloc(&d_G.elements,size);	
	//create handle for Cublas
	cublasHandle_t handle;
	cublasCreate(&handle);	
		
	for (i = 0; i < max_iter; i++){
			// compute G
			Matrix_x_Matrix(handle,d_A,d_Series_sub1,d_Temp);
			Matrix_minus_Matrix(handle,d_Temp,d_Series_sub2,d_Temp,1);
			Matrix_x_Matrix(handle,d_Temp,d_Series_sub1_Transpose,d_G);
			// compute the A[ind] in the next iteratio
			Matrix_minus_Matrix(handle,d_A,d_G,d_A,delta);
	}

	cublasDestroy(handle);
	// Free the tempory memomry
	free(Series_sub1_Transpose.elements);
	free(Series_sub2.elements);
	free(Series_sub1.elements);
	cudaFree(d_G.elements);
	cudaFree(d_Temp.elements);
	cudaFree(d_A.elements);
	cudaFree(d_Series_sub1_Transpose.elements);
	cudaFree(d_Series_sub2.elements);
	cudaFree(d_Series_sub1.elements);
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
	int N=nRow;
	int T = nCol;
	
	Matrix Series;
	Series.width = 	nCol;
	Series.height = nRow;
	size = Series.width*Series.height*sizeof(double);
	Series.elements = (double*)malloc(size);
	for(row = 0; row < nRow; row++) {
		for(col = 0; col < nCol; col++) {
			Series.elements[row+col*Series.height] = Series_Pr[nRow * col + row];// this needs testing
		}
	}
	
	int i,j;
	Matrix A;
	//create matrix A
	A.width = N;
	A.height = N;
	size = A.width*A.height*sizeof(double);
	A.elements = (double*)malloc(size);
	//time counter
	double tstart,tstop,ttime;
	
	tstart =(double)clock()/CLOCKS_PER_SEC;
	int counter = 0;
	while(counter<10){
		//initialize A	
		for(i =0; i < A.height; i++){
			for(j = 0; j < A.width; j++){
				A.elements[i+j*A.height] = 0;
			}
		}
		
		// Gradient_descent
		Gradient_descent(A,Series,max_iter,delta,N,T);
		counter++;
	}
	tstop = (double)clock()/CLOCKS_PER_SEC;
	ttime=tstop-tstart;
	printf("time:%fs\n",ttime);
	free(A.elements);
	free(Series.elements);
	return 0;
}
