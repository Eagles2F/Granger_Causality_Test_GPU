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
void Matrix_minus_Matrix(cublasHandle_t &handle,Matrix A,Matrix B, Matrix C,double alpha,double beta){
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

void Fun(Matrix A, Matrix B, double Lamda){
	int i,j;
	for(i=0;i++;i<A.height){
		for(j=0;j++;j<A.width){
			B.elements[i+j*B.height]=(abs(A.elements[i+j*A.height])>Lamda)*(abs(A.elements[i+j*A.height]-Lamda))*(A.elements[i+j*A.height]>0?1:-1);
		}
	}
}

void Lasso_Granger(Matrix A, Matrix Series,int max_iter, double delta,int N,int T,double Lamda){
	
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


	//allocate the device memory
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

	Matrix d_Series00;
	d_Series00.width = N;
	d_Series00.height = N;
	size = d_Series00.width*d_Series00.height*sizeof(double);
	cudaMalloc(&d_Series00.elements,size);

	Matrix d_Series10;
	d_Series10.width = N;
	d_Series10.height = N;
	size = d_Series10.width*d_Series10.height*sizeof(double);
	cudaMalloc(&d_Series10.elements,size);

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
	
	Matrix d_YA;
	d_YA.width =A.width;
	d_YA.height = A.height;
	size = d_YA.width*d_YA.height*sizeof(double);
	cudaMalloc(&d_YA.elements,size);
	cudaMemcpy(d_YA.elements,A.elements,size,cudaMemcpyHostToDevice);
	
	Matrix A_new;
	A_new.width =A.width;
	A_new.height = A.height;
	size = A_new.width*A_new.height*sizeof(double);
	A_new.elements =(double*) malloc(size);

	Matrix d_A_new;
	d_A_new.width =A.width;
	d_A_new.height = A.height;
	size = d_A_new.width*d_A_new.height*sizeof(double);
	cudaMalloc(&d_A_new.elements,size);
	cudaMemcpy(d_A_new.elements,A.elements,size,cudaMemcpyHostToDevice);
	
	Matrix d_Temp;
	d_Temp.width = N;
	d_Temp.height = N;
	size = d_Temp.width*d_Temp.height*sizeof(double);
	cudaMalloc(&d_Temp.elements,size);

	Matrix d_G;
	d_G.width = N;
	d_G.height= N;
	size = d_G.width*d_G.height*sizeof(double);
	cudaMalloc(&d_G.elements,size);	

	Matrix G;
	G.width = N;
	G.height= N;
	size = G.width*G.height*sizeof(double);
	G.elements = (double*)malloc(size);	


	double t =1;
	double t_new;
	//create handle for Cublas
	cublasHandle_t handle;
	cublasCreate(&handle);	
		
	//compute d_Series00 & d_Series10
	Matrix_x_Matrix(handle,d_Series_sub1,d_Series_sub1_Transpose,d_Series00);
	Matrix_x_Matrix(handle,d_Series_sub2,d_Series_sub1_Transpose,d_Series10);

	for (i = 0; i < max_iter; i++){
			// compute G
			Matrix_x_Matrix(handle,d_YA,d_Series00,d_Temp);
			Matrix_minus_Matrix(handle,d_Temp,d_Series10,d_Temp,1,-1);
			Matrix_minus_Matrix(handle,d_YA,d_Temp,d_G,1,-delta);
			
			size = G.width*G.height*sizeof(double);	
			cudaMemcpy(G.elements,d_G.elements,size,cudaMemcpyDeviceToHost);
		
			//compute S_new
			Fun(G,A_new,Lamda);
			size = A_new.width*A_new.height*sizeof(double);
			cudaMemcpy(d_A_new.elements,A_new.elements,size,cudaMemcpyHostToDevice);	
			//AGP updates
			t_new = (1+sqrt(1+4*t*t))/2;
			Matrix_minus_Matrix(handle,d_A_new,d_A,d_YA,(t+t_new-1)/t_new,(1-t)/t_new);
			//Variable updates
			Matrix_minus_Matrix(handle,d_A_new,d_A,d_A,1,0);
			t = t_new;
	}

	cublasDestroy(handle);
	// Free the tempory memomry
	free(G.elements);
	free(A_new.elements);
	free(Series_sub1_Transpose.elements);
	free(Series_sub2.elements);
	free(Series_sub1.elements);
	cudaFree(d_G.elements);
	cudaFree(d_Temp.elements);
	cudaFree(d_A_new.elements);
	cudaFree(d_YA.elements);
	cudaFree(d_A.elements);
	cudaFree(d_Series_sub1_Transpose.elements);
	cudaFree(d_Series10.elements);
	cudaFree(d_Series00.elements);
	cudaFree(d_Series_sub2.elements);
	cudaFree(d_Series_sub1.elements);
}
	
int main(int argc, char** argv){
	double Lamda = (double)pow(10,-9);
	double delta = (double)pow(10,-3);
	int max_iter = 500;
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
		Lasso_Granger(A,Series,max_iter,delta,N,T,Lamda);
		counter++;
	}
	tstop = (double)clock()/CLOCKS_PER_SEC;
	ttime=tstop-tstart;
	printf("time:%fs\n",ttime);
	free(A.elements);
	free(Series.elements);
	return 0;
}
