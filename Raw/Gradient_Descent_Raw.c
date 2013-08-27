#include <stdio.h>
#include <math.h>
#include <time.h>
#include "mat.h"

// Matrices are stored in row-major order: 
// M(row, col) = *(M.elements + row * M.width + col) 
typedef struct {
    	int width;    
 	int height;    
	double* elements; 
	}Matrix;

//Vector * Matrix
void Vec_x_Matrix(double A[],Matrix B,double C[],int N,int M){
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
void Vec_sub(double A[],double B[],double C[],int N){
	int i;
	for(i = 0; i < N; i++){
		C[i] = A[i]-B[i];
	}
}

//Constant x Vector
void Con_x_Vector(double Con, double * Vec, int N,double* Vec_R){
	int i;
	for(i=0;i<N;i++){
		Vec_R[i]=Vec[i]*Con;
	}
}

//Get sub_matrix from matrix
void Sub_Matrix(Matrix m,Matrix Sub_m,int N,int M,int start,int end){
	int i,j;
	for(i=0; i< N; i++){
		for(j=start; j <end+1; j++){
			Sub_m.elements[i*Sub_m.width+j-start] = m.elements[i*m.width+j];
		}
	}
}

void Gradient_descent(Matrix A, Matrix Series,int max_iter, double delta,int N,int T){
// create the vector for the intermediate result
	double* G = (double*)malloc(N*sizeof(double)); 
	double* Temp = (double*)malloc((T-1)*sizeof(double));
	double* Aind = (double*)malloc(N*sizeof(double));
	double* Series_sub2ind = (double*)malloc((T-1)*sizeof(double));
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
			Series_sub1_Transpose.elements[i*Series_sub1_Transpose.width+j]=Series_sub1.elements[j*Series_sub1.height+i];
		}
	}

	for(ind=0; ind<N; ind++){
		for(i =0; i < N; i++){
			Aind[i]=A.elements[ind*A.width+i];
		}
		for(i=0; i < T-1; i++){
			Series_sub2ind[i]=Series_sub2.elements[ind*Series_sub2.width+i];
		}
		for (i = 0; i < max_iter; i++){
			// compute G
			Vec_x_Matrix(Aind,Series_sub1,Temp,N,T-1);
			Vec_sub(Temp,Series_sub2ind,Temp,T-1);
			Vec_x_Matrix(Temp,Series_sub1_Transpose,G,T-1,N);
	
			// compute the A[ind] in the next iteration
			Con_x_Vector(delta,G,N,G);
			Vec_sub(Aind,G,Aind,N);
		}
		//copy Aind to A[ind][:]
		for(i= 0; i<N;i++) {
			A.elements[ind*A.width+i] = Aind[i];
		}
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
	int T = 800;
	int N = 100;
	double delta = (double)pow(10,-5);
	int max_iter = 100;
	size_t size;
	
	//load Series from .mat file
	MATFile *pmat;	
	const char* file ="Series_100_800.mat";
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
				A.elements[i*A.width+j] = 0;
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
