// Solving a linear system involving the Hilbert matrix using the CG method and CBLAS
// Author: Petter Frisvåg

#include <iostream>
#include <cstdlib>
#include <cblas.h>

template <typename T>
T** mk_2D_array(size_t m, size_t n);

template <typename T>
void del_2D_array(T **arr);

int main(int argc, char **argv) {
	if (argc != 2) {
		std::cout << "expects one argument n\n";
		return EXIT_FAILURE;
	}
	size_t n = atoi(argv[1]);

	double *b = new double[n];
	double **A = mk_2D_array<double>(n,n);

	double *x_k = new double[n];
	double *r_k = new double[n];
	double *p_k = new double[n];
	
	double *r_k_p = new double[n];
	double *Ap_k = new double[n];

	for (size_t i = 0; i < n; i++) {
		for (size_t j = i; j < n; j++) {
			A[i][j] = A[j][i] = 1.0/(i+j+1);
		}
		b[i] = 1;
		x_k[i] = 0;
	}
	cblas_dcopy(n,b,1,r_k,1);
	cblas_dgemv(CblasRowMajor,CblasNoTrans,n,n,1,*A,n,x_k,1,-1,r_k,1);
	cblas_dcopy(n,r_k,1,p_k,1);
	cblas_dscal(n,-1,p_k,1);

	size_t k;
	double alpha_k, beta_k;
	for (k = 0; k < 1000; k++) {
		if (cblas_dnrm2(n,r_k,1) < 1e-6) {
			break;
		}
		cblas_dgemv(CblasRowMajor,CblasNoTrans,n,n,1,*A,n,p_k,1,0,Ap_k,1);
		alpha_k = cblas_ddot(n,r_k,1,r_k,1)/cblas_ddot(n,p_k,1,Ap_k,1);
		cblas_daxpy(n,alpha_k,p_k,1,x_k,1);
		cblas_dcopy(n,r_k,1,r_k_p,1);
		cblas_daxpy(n,alpha_k,Ap_k,1,r_k,1);
		beta_k = cblas_ddot(n,r_k,1,r_k,1)/cblas_ddot(n,r_k_p,1,r_k_p,1);
		cblas_dscal(n,beta_k,p_k,1);
		cblas_daxpy(n,-1,r_k,1,p_k,1);
	}
	//cblas_dgemv(CblasRowMajor,CblasNoTrans,n,n,alpha,*A,n,x,1,beta,b,1);
	std::cout << "k = " << k << ", x_k = (";
	for (size_t i = 0; i < n-1; i++) {
		std::cout << x_k[i] << ",";
	}
	std::cout << x_k[n-1] << ")" << std::endl;

	delete [] x_k;
	delete [] p_k;
	delete [] r_k_p;
	delete [] Ap_k;
	delete [] b;
	del_2D_array<double>(A);

	return EXIT_SUCCESS;
}


// source: http://stackoverflow.com/a/21944048
template <typename T>
T** mk_2D_array(size_t m, size_t n) {
   T** ptr = new T*[m];  // allocate pointers
   T* pool = new T[m*n];  // allocate pool
   for (size_t i = 0; i < m; ++i, pool += n)
       ptr[i] = pool;
   return ptr;
}

template <typename T>
void del_2D_array(T **arr) {
   delete [] arr[0];  // remove the pool
   delete [] arr;     // remove the pointers
}