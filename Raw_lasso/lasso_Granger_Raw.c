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

// Matrix x Matrix x delta
void Matrix_x_Matrix(Matrix A,Matrix B,Matrix C,double delta){
	int i,j,k;
	double  sum=0;
	for(i=0;i<C.height;i++){
		for(j=0;j<C.width;j++){
			for(k=0;k<A.width;k++){
				sum=sum+A.elements[i*A.width+k]*B.elements[k*B.width+j];
			}
			C.elements[i*C.width+j]=sum*delta;
			sum=0;
		}
	}
}

//Matrix*alpha+Matrix*delta
void Matrix_add_Matrix(Matrix A,Matrix B,Matrix C,double alpha,double delta){
	int i,j;
	for(i=0;i<A.height;i++){
		for(j=0;j<A.width;j++){
			C.elements[i*C.width+j]=alpha*A.elements[i*A.width+j]+delta*B.elements[i*B.width+j];
		}
	}
}

//Matrix value copy A <= B
void Copy_Matrix_Value(Matrix A, Matrix B){
	int i,j;
	for(i=0;i<A.height;i++){
		for(j=0;j<A.width;j++){
			A.elements[i*A.width+j]=B.elements[i*B.width+j];
		}
	}
}

// get the absolute value of double a
double absolute(double a){
	if(a>=0){
		return a;
	}
	else{
		return -a;
	}	
}
// get the sign of a
double sign(double a){
	if(a >0){
		return 1;
	}
	else if(a ==0){
		return 0;
	}
	else if(a < 0){
		return -1;
	}
}

//Fun	
void Fun(Matrix A,Matrix B,double Lamda){
	int i,j;
	for(i=0;i<A.height;i++){
		for(j=0;j<A.width;j++){
			B.elements[i*B.width+j]=(absolute(A.elements[i*A.width+j])>Lamda)*(absolute(A.elements[i*A.width+j])-Lamda)*sign(A.elements[i*A.width+j]);
		}
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

void Lasso_Granger(Matrix A, Matrix Series,int max_iter, double delta,int N,int T, double Lamda){

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
			Series_sub1_Transpose.elements[i*Series_sub1_Transpose.width+j]=Series_sub1.elements[j*Series_sub1.width+i];
		}
	}

	// allocate the space for Series00
	Matrix Series00;
	Series00.width = N;
	Series00.height = N;
	size = Series00.width*Series00.height*sizeof(double);
	Series00.elements = (double*)malloc(size);

	// allocate the space for Series10
	Matrix Series10;
	Series10.width = N;
	Series10.height = N;
	 size = Series10.width*Series10.height*sizeof(double);
	Series10.elements = (double*)malloc(size);

	// allocate the space for YA
	Matrix YA;
	YA.width = N;
	YA.height =N;
	size = YA.width*YA.height*sizeof(double);
	YA.elements = (double*)malloc(size);

	// allocate the space for Temp
	Matrix Temp;
	Temp.width = N;
	Temp.height = N;
	size = Temp.width*Temp.height*sizeof(double);
	Temp.elements = (double*)malloc(size);

	// allocate the space for G
	Matrix G;
	G.width = N;
	G.height = N;
	size = G.height*G.width*sizeof(double);
	G.elements = (double*)malloc(size);

	// allocate the space for A_new
	Matrix A_new;
	A_new.width =N;
	A_new.height =N;
	size=A_new.height*A_new.width*sizeof(double);
	A_new.elements = (double*)malloc(size);
	
	//Compute Series00 & Series10
	Matrix_x_Matrix(Series_sub1,Series_sub1_Transpose,Series00,1.0/T);
	Matrix_x_Matrix(Series_sub2,Series_sub1_Transpose,Series10,1.0/T);

	double t =1 ;
	double t_new;


	for (i = 0; i < max_iter; i++){
		// compute G
		Matrix_x_Matrix(YA,Series00,Temp,1);
		Matrix_add_Matrix(Temp,Series10,Temp,1,-1);
		Matrix_add_Matrix(YA,Temp,G,1,-delta);
		
		//Compute A_new
		Fun(G,A_new,Lamda);
		
		//AGP updates
		t_new = (1+sqrt(1+4*t*t))/2;
		Matrix_add_Matrix(A_new,A,YA,(t+t_new-1.0)/t_new,(1.0-t)/t_new);
		
		//Varable updates
		Copy_Matrix_Value(A,A_new);
		t = t_new;
	}
		

	// Free the tempory memomry
	free(A_new.elements);
	free(G.elements);
	free(Temp.elements);
	free(YA.elements);
	free(Series10.elements);
	free(Series00.elements);
	free(Series_sub1_Transpose.elements);
	free(Series_sub2.elements);
	free(Series_sub1.elements);
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
	int T = nCol;
	int N = nRow;
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
		
		// Lasso_Granger
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
