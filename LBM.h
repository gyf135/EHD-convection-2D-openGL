/* This code accompanies
 *   Two relaxation time lattice Boltzmann method coupled to fast Fourier transform Poisson solver: Application to electroconvective flow, Journal of Computational Physics
 *	 https://doi.org/10.1016/j.jcp.2019.07.029
 *   Numerical analysis of electroconvection in cross-flow with unipolar charge injection, Physical Review Fluids
 *	 https://doi.org/10.1103/PhysRevFluids.4.103701
 *   Yifei Guan, Igor Novosselov
 * 	 University of Washington
 *
 * Author: Yifei Guan
 *
 */
#ifndef __LBM_H
#define __LBM_H
#include <math.h>
#include <cufft.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int iteractionCount = 0;
double *T = (double*)malloc(sizeof(double));
double *M = (double*)malloc(sizeof(double));
double *C = (double*)malloc(sizeof(double));
double *Fe = (double*)malloc(sizeof(double));

const unsigned int flag = 1; // if flat == 1, read previous data, otherwise initialize
const int nThreads = 61; // can divide NX

// define grids
const unsigned int NX = 122; // number of grid points in x-direction, meaning 121 cells while wavelength is 122 with periodic boundaries
const unsigned int W  = NX;
const unsigned int NY = 101; // number of grid points in y-direction, meaning NY-1 cells
const unsigned int H = NY;
const unsigned int NE = 2 * (NY - 1);
const unsigned int size = NX*NE;
__constant__ double LL = 1.22;
__constant__ double Lx = 1.22;
__constant__ double Ly = 1.0;
__constant__ double dx = 1.0 / 100.0; //need to change according to NX and LX
__constant__ double dy = 1.0 / 100.0; //need to change according to NY and LY

// define physics
double uw_host = 0.0; // velocity of the wall
double exf_host = 0.0; // external force for poisseuille flow
__device__ double uw;
__device__ double exf;
__constant__ double CFL = 0.05; // CFL = dt/dx
__constant__ double dt = 0.05*1.0 / 100.0; // dt = dx * CFL need to change according to dx, dy
__constant__ double cs_square = 1.0 / 3.0 / (0.05*0.05); // 1/3/(CFL^2)
__constant__ double rho0 = 1600.0;
__device__ double charge0;
double charge0_host = 10.0;
__device__ double voltage;
double voltage_host = 1.0e4;
__constant__ double eps = 1.0e-4;
__constant__ double K = 2.5e-5;
double K_host;
__constant__ double diffu = 6.25e-5;
double nu_host = 0.08;
__device__ double nu;
int perturb_host = 0;
__device__ int perturb;

// define scheme
const unsigned int ndir = 9;
const size_t mem_size_0dir = sizeof(double)*NX*NY;
const size_t mem_size_n0dir = sizeof(double)*NX*NY*(ndir - 1);
const size_t mem_size_scalar = sizeof(double)*NX*NY;
const size_t mem_size_ext_scalar = sizeof(double)*NX*NE;

// weights of populations (total 9 for D2Q9 scheme)
__constant__ double w0 = 4.0 / 9.0;  // zero weight
__constant__ double ws = 1.0 / 9.0;  // adjacent weight
__constant__ double wd = 1.0 / 36.0; // diagonal weight

// parameters for (two-relaxation time) TRT scheme
__constant__ double V  = 1.0 / 12.0;
__constant__ double VC = 1.0e-6;

const unsigned int NSTEPS = 500;
const unsigned int NSAVE = 5;// NSTEPS / 4 + 1;
const unsigned int NMSG   =  NSAVE;

// physical time
double t = 0.0;


double parameter = 0.0;
double X = 0.0;

double *f0_gpu, *f1_gpu, *f2_gpu;
double *h0_gpu, *h1_gpu, *h2_gpu;
double *rho_gpu, *ux_gpu, *uy_gpu;
double *charge_gpu, *phi_gpu;
double *Ex_gpu, *Ey_gpu;
double *kx, *ky;
cufftHandle plan = 0;
cufftDoubleComplex *freq_gpu_ext, *charge_gpu_ext, *phi_gpu_ext;
double *f0bc; // store f0 of the lower plate for further use
double *kx_host = (double*)malloc(sizeof(double)*NX);
double *ky_host = (double*)malloc(sizeof(double)*NE);
double dt_host;
double Lx_host;
double Ly_host;
double dy_host;
double *charge_host = (double*)malloc(mem_size_scalar);
double *Ey_host     = (double*)malloc(mem_size_scalar);

// suppress verbose output
const bool quiet = true;

void initialization(double*, double*, double*, double*, double*, double*, double*);
void stream_collide_save(double*,double*,double*,double*,double*,double*, double*, double*, double*, double*, double*,double*,double,double*);
void init_equilibrium(double*,double*,double*,double*,double*, double*, double*, double*, double*, double*);
void report_flow_properties(unsigned int,double,double*,double*,double*,double*, double*, double*, double*);
void save_scalar(const char*,double*,double*,unsigned int);
void save_data_tecplot(double, double*, double*, double*, double*, double*, double*, double*);
void compute_parameters(double*, double*, double*, double*);
void poisson_phi(double*, double*);
void extension(double*, cufftDoubleComplex*);
void efield(double*, double*, double*);
void derivative(double*, double*, cufftDoubleComplex*);
void extract(double*, cufftDoubleComplex*);
void kernelLauncher();
//void perturbFlow();
double current(double*, double*);

inline size_t scalar_index(unsigned int x, unsigned int y)
{
	return NX*y + x;
}

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}
#endif /* __LBM_H */

