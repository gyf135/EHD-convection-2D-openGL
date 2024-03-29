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
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda.h>
#include "LBM.h"
#include <cuda_runtime.h>


__device__ __forceinline__ size_t gpu_field0_index(unsigned int x, unsigned int y)
{
    return NX*y+x;
}

__device__ __forceinline__ size_t gpu_scalar_index(unsigned int x, unsigned int y)
{
    return NX*y+x;
}

__device__ __forceinline__ size_t gpu_fieldn_index(unsigned int x, unsigned int y, unsigned int d)
{
    return (NX*(NY*(d-1)+y)+x);
}

#define checkCudaErrors(err)  __checkCudaErrors(err,#err,__FILE__,__LINE__)
#define getLastCudaError(msg)  __getLastCudaError(msg,__FILE__,__LINE__)

inline void __checkCudaErrors(cudaError_t err, const char *const func, const char *const file, const int line )
{
    if(err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s(%d)\"%s\": [%d] %s.\n",
                file, line, func, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __getLastCudaError(const char *const errorMessage, const char *const file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s(%d): [%d] %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

// forward declarations of kernels
__global__ void gpu_initialization(double*, double*, double*, double*, double*, double*, double*);
//__global__ void gpu_taylor_green(unsigned int,double*,double*,double*);
__global__ void gpu_init_equilibrium(double*,double*,double*,double*,double*, double*, double*, double*, double*, double*);
__global__ void gpu_collide_save(double*,double*,double*,double*,double*,double*, double*, 
	double*, double*, double*, double*, double*, double,double*);
__global__ void gpu_boundary(double*, double*, double*, double*, double*, double*,double*);
__global__ void gpu_stream(double*, double*, double*, double*, double*, double*);
__global__ void gpu_bc_charge(double*, double*, double*);

/*
__device__ void taylor_green_eval(unsigned int t, unsigned int x, unsigned int y, double *r, double *u, double *v)
{
    double kx = 2.0*M_PI/NX;
    double ky = 2.0*M_PI/NY;
    double td = 1.0/(nu*(kx*kx+ky*ky));
    
    double X = x+0.5;
    double Y = y+0.5;
    double ux = -u_max*sqrt(ky/kx)*cos(kx*X)*sin(ky*Y)*exp(-1.0*t/td);
    double uy =  u_max*sqrt(kx/ky)*sin(kx*X)*cos(ky*Y)*exp(-1.0*t/td);
    double P = -0.25*rho0*u_max*u_max*((ky/kx)*cos(2.0*kx*X)+(kx/ky)*cos(2.0*ky*Y))*exp(-2.0*t/td);
    double rho = rho0+3.0*P;
    
    *r = rho;
    *u = ux;
    *v = uy;
}
*/
__host__ void initialization(double *r, double *c, double *fi, double *u, double *v, double *ex, double *ey)
{
	// blocks in grid
	dim3 grid(NX / nThreads, NY, 1);

	// threads in block
	dim3 threads(nThreads, 1, 1);

	gpu_initialization << <grid, threads >> > (r, c, fi, u, v, ex, ey);
	getLastCudaError("gpu_taylor_green kernel error");
}

__global__ void gpu_initialization(double *r, double *c, double *fi, double *u, double *v, double *ex, double *ey)
{
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	size_t sidx = gpu_scalar_index(x, y);
	r[sidx]  = rho0;
	c[sidx]  = 0.0;
	fi[sidx] = voltage * (Ly - dy*y) / Ly;
	u[sidx]  = 0.0;
	v[sidx]  = 0.0;
	ex[sidx] = 0.0;
	ey[sidx] = 0.0;
}

__host__ void init_equilibrium(double *f0, double *f1, double *h0, double *h1, double *r, double *c, 
								double *u, double *v, double *ex, double *ey)
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, 1);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_init_equilibrium<<< grid, threads >>>(f0,f1,h0,h1,r,c,u,v,ex,ey);
    getLastCudaError("gpu_init_equilibrium kernel error");
}

__global__ void gpu_init_equilibrium(double *f0, double *f1, double *h0, double *h1, double *r, double *c,
										double *u, double *v, double *ex, double *ey)
{
    unsigned int y = blockIdx.y;
    unsigned int x = blockIdx.x*blockDim.x+threadIdx.x;
    
    double rho    = r[gpu_scalar_index(x,y)];
    double ux     = u[gpu_scalar_index(x,y)];
    double uy     = v[gpu_scalar_index(x,y)];
	double charge = c[gpu_scalar_index(x, y)];
	double Ex     = ex[gpu_scalar_index(x, y)];
	double Ey     = ey[gpu_scalar_index(x, y)];

    // load equilibrium
    // feq_i  = w_i rho [1 + 3(ci . u) + (9/2) (ci . u)^2 - (3/2) (u.u)]
    // feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u) + (1/2) (ci . 3u)^2]
    // feq_i  = w_i rho [1 - 3/2 (u.u) + (ci . 3u){ 1 + (1/2) (ci . 3u) }]
    
    // temporary variables
    double w0r = w0*rho;
    double wsr = ws*rho;
    double wdr = wd*rho;
	double w0c = w0*charge;
	double wsc = ws*charge;
	double wdc = wd*charge;

    double omusq   = 1.0 - 0.5*(ux*ux+uy*uy)/cs_square;
	double omusq_c = 1.0 - 0.5*((ux + K*Ex)*(ux + K*Ex) + (uy + K*Ey)*(uy + K*Ey)) / cs_square;
    
    double tux   = ux / cs_square / CFL;
    double tuy   = uy / cs_square / CFL;
	double tux_c = (ux + K*Ex) / cs_square / CFL;
	double tuy_c = (uy + K*Ey) / cs_square / CFL;
    
	// zero weight
    f0[gpu_field0_index(x,y)]    = w0r*(omusq);
	h0[gpu_field0_index(x,y)]    = w0c*(omusq_c);
    
	// adjacent weight
	// flow
    double cidot3u = tux;
    f1[gpu_fieldn_index(x,y,1)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuy;
    f1[gpu_fieldn_index(x,y,2)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tux;
    f1[gpu_fieldn_index(x,y,3)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -tuy;
    f1[gpu_fieldn_index(x,y,4)]  = wsr*(omusq + cidot3u*(1.0+0.5*cidot3u));
	// charge
	cidot3u = tux_c;
	h1[gpu_fieldn_index(x, y, 1)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c;
	h1[gpu_fieldn_index(x, y, 2)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c;
	h1[gpu_fieldn_index(x, y, 3)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c;
	h1[gpu_fieldn_index(x, y, 4)] = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
    
	// diagonal weight
	// flow
    cidot3u = tux+tuy;
    f1[gpu_fieldn_index(x,y,5)]  = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tuy-tux;
    f1[gpu_fieldn_index(x,y,6)]  = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = -(tux+tuy);
    f1[gpu_fieldn_index(x,y,7)]  = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));
    cidot3u = tux-tuy;
    f1[gpu_fieldn_index(x,y,8)]  = wdr*(omusq + cidot3u*(1.0+0.5*cidot3u));

	// charge
	cidot3u = tux_c + tuy_c;
	h1[gpu_fieldn_index(x, y, 5)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tux_c;
	h1[gpu_fieldn_index(x, y, 6)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -(tux_c + tuy_c);
	h1[gpu_fieldn_index(x, y, 7)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuy_c;
	h1[gpu_fieldn_index(x, y, 8)] = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
}

__host__ void stream_collide_save(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *r, double *c, 
	double *u, double *v, double *ex, double *ey,double t,double *f0bc)
{
    // blocks in grid
    dim3  grid(NX/nThreads, NY, 1);
    // threads in block
    dim3  threads(nThreads, 1, 1);

    gpu_collide_save<<< grid, threads >>>(f0,f1,f2, h0, h1, h2, r, c, u,v, ex, ey,t,f0bc);
	gpu_boundary << < grid, threads >> >(f0, f1, f2, h0, h1, h2, f0bc);
	gpu_stream << < grid, threads >> >(f0, f1, f2, h0, h1, h2);
	gpu_bc_charge << < grid, threads >> >(h0, h1, h2);


    getLastCudaError("gpu_stream_collide_save kernel error");
}

__global__ void gpu_collide_save(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *r, double *c,
	double *u, double *v, double *ex, double *ey, double t,double *f0bc)
{
	// useful constants
	double omega_plus = 1.0 / (nu / cs_square / dt + 1.0 / 2.0) / dt;
	double omega_minus = 1.0 / (V / (nu / cs_square / dt) + 1.0 / 2.0) / dt;
	double omega_c_minus = 1.0 / (diffu / cs_square / dt + 1.0 / 2.0) / dt;
	double omega_c_plus = 1.0 / (VC / (diffu / cs_square / dt) + 1.0 / 2.0) / dt;

	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	// storage of f0 at upper and lower plate
	if (y == 0) f0bc[gpu_field0_index(x, 0)]  = f0[gpu_field0_index(x, 0)];    // lower plate
	
	if (y==NY-1) f0bc[gpu_field0_index(x, 1)] = f0[gpu_field0_index(x, NY - 1)]; // upper plate

	// load populations from nodes (ft is the same as f1)
	double ft0 = f0[gpu_field0_index(x, y)];
	double ht0 = h0[gpu_field0_index(x, y)];
	double ft1 = f1[gpu_fieldn_index(x, y, 1)];
	double ft2 = f1[gpu_fieldn_index(x, y, 2)];
	double ft3 = f1[gpu_fieldn_index(x, y, 3)];
	double ft4 = f1[gpu_fieldn_index(x, y, 4)];
	double ft5 = f1[gpu_fieldn_index(x, y, 5)];
	double ft6 = f1[gpu_fieldn_index(x, y, 6)];
	double ft7 = f1[gpu_fieldn_index(x, y, 7)];
	double ft8 = f1[gpu_fieldn_index(x, y, 8)];
	double ht1 = h1[gpu_fieldn_index(x, y, 1)];
	double ht2 = h1[gpu_fieldn_index(x, y, 2)];
	double ht3 = h1[gpu_fieldn_index(x, y, 3)];
	double ht4 = h1[gpu_fieldn_index(x, y, 4)];
	double ht5 = h1[gpu_fieldn_index(x, y, 5)];
	double ht6 = h1[gpu_fieldn_index(x, y, 6)];
	double ht7 = h1[gpu_fieldn_index(x, y, 7)];
	double ht8 = h1[gpu_fieldn_index(x, y, 8)];

	// compute macroscopic variables from microscopic variables
	double rho = ft0 + ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + ft7 + ft8;
	double rhoinv = 1.0 / rho;
	double charge = ht0 + ht1 + ht2 + ht3 + ht4 + ht5 + ht6 + ht7 + ht8;
	double Ex = ex[gpu_scalar_index(x, y)];
	double Ey = ey[gpu_scalar_index(x, y)];
	double forcex = charge * Ex + exf;
	double forcey = charge * Ey;
	double ux = rhoinv*((ft1 + ft5 + ft8 - (ft3 + ft6 + ft7)) / CFL + forcex*dt*0.5);
	double uy = rhoinv*((ft2 + ft5 + ft6 - (ft4 + ft7 + ft8)) / CFL + forcey*dt*0.5);
	if (perturb==1){
		double xx = x*dx;
		double yy = y*dy;
		uy = (cos(2.0 * M_PI*yy) - 1.0)*cos(2.0 * M_PI / Lx*xx)*0.001;
		ux = Lx*sin(2.0*M_PI*yy)*sin(2.0*M_PI / Lx*xx)*0.001;
	}
	else{
		if (y == 0) {
			double ftm0 = f0[gpu_field0_index(x, 1)];
			double htm0 = h0[gpu_field0_index(x, 1)];
			double ftm1 = f1[gpu_fieldn_index(x, 1, 1)];
			double ftm2 = f1[gpu_fieldn_index(x, 1, 2)];
			double ftm3 = f1[gpu_fieldn_index(x, 1, 3)];
			double ftm4 = f1[gpu_fieldn_index(x, 1, 4)];
			double ftm5 = f1[gpu_fieldn_index(x, 1, 5)];
			double ftm6 = f1[gpu_fieldn_index(x, 1, 6)];
			double ftm7 = f1[gpu_fieldn_index(x, 1, 7)];
			double ftm8 = f1[gpu_fieldn_index(x, 1, 8)];
			double htm1 = h1[gpu_fieldn_index(x, 1, 1)];
			double htm2 = h1[gpu_fieldn_index(x, 1, 2)];
			double htm3 = h1[gpu_fieldn_index(x, 1, 3)];
			double htm4 = h1[gpu_fieldn_index(x, 1, 4)];
			double htm5 = h1[gpu_fieldn_index(x, 1, 5)];
			double htm6 = h1[gpu_fieldn_index(x, 1, 6)];
			double htm7 = h1[gpu_fieldn_index(x, 1, 7)];
			double htm8 = h1[gpu_fieldn_index(x, 1, 8)];

			// compute macroscopic variables from microscopic variables
			double rhom = ftm0 + ftm1 + ftm2 + ftm3 + ftm4 + ftm5 + ftm6 + ftm7 + ftm8;
			double rhoinvm = 1.0 / rhom;
			double chargem = htm0 + htm1 + htm2 + htm3 + htm4 + htm5 + htm6 + htm7 + htm8;
			double Exm = ex[gpu_scalar_index(x, 1)];
			double Eym = ey[gpu_scalar_index(x, 1)];
			double forcexm = charge * Ex + exf;
			double forceym = charge * Ey;

			ux = -rhoinvm*((ftm1 + ftm5 + ftm8 - (ftm3 + ftm6 + ftm7)) / CFL + forcexm*dt*0.5);
			uy = -rhoinvm*((ftm2 + ftm5 + ftm6 - (ftm4 + ftm7 + ftm8)) / CFL + forceym*dt*0.5);
		}
	}
	

	// write to memory (only when visualizing the data)
	
	r[gpu_scalar_index(x, y)] = rho;
	u[gpu_scalar_index(x, y)] = ux;
	v[gpu_scalar_index(x, y)] = uy;
	c[gpu_scalar_index(x, y)] = charge;

	// collision step
	// now compute and relax to equilibrium
	// note that
	// feq_i  = w_i rho [1 + (ci . u / cs_square) + (1/2) (ci . u / cs_square)^2 - (1/2) (u.u) / cs_square]
	// feq_i  = w_i rho [1 - 1/2 (u.u)/cs_square + (ci . u / cs_square) + (1/2) (ci . u / cs_square)^2]
	// feq_i  = w_i rho [1 - 1/2 (u.u)/cs_square + (ci . u/cs_square){ 1 + (1/2) (ci . u/cs_square) }]
	// for charge transport equation, just change u into u + KE
	// heq_i  = w_i charge [1 - 1/2 (u.u)/cs_square + (ci . u/cs_square){ 1 + (1/2) (ci . u/cs_square) }]

	// choices of c
	// cx = [0, 1, 0, -1, 0, 1, -1, -1, 1] / CFL
	// cy = [0, 0, 1, 0, -1, 1, 1, -1, -1] / CFL

	// calculate equilibrium
	// temporary variables
	double w0r = w0*rho;
	double wsr = ws*rho;
	double wdr = wd*rho;
	double w0c = w0*charge;
	double wsc = ws*charge;
	double wdc = wd*charge;

	double omusq = 1.0 - 0.5*(ux*ux + uy*uy) / cs_square;
	double omusq_c = 1.0 - 0.5*((ux + K*Ex)*(ux + K*Ex) + (uy + K*Ey)*(uy + K*Ey)) / cs_square;

	double tux = ux / cs_square / CFL;
	double tuy = uy / cs_square / CFL;
	double tux_c = (ux + K*Ex) / cs_square / CFL;
	double tuy_c = (uy + K*Ey) / cs_square / CFL;

	// zero weight
	double fe0 = w0r*(omusq);
	double he0 = w0c*(omusq_c);

	// adjacent weight
	// flow
	double cidot3u = tux;
	double fe1 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy;
	double fe2 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux;
	double fe3 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy;
	double fe4 = wsr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	// charge
	cidot3u = tux_c;
	double he1 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c;
	double he2 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tux_c;
	double he3 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -tuy_c;
	double he4 = wsc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

	// diagonal weight
	// flow
	cidot3u = tux + tuy;
	double fe5 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy - tux;
	double fe6 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -(tux + tuy);
	double fe7 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux - tuy;
	double fe8 = wdr*(omusq + cidot3u*(1.0 + 0.5*cidot3u));
	// charge
	cidot3u = tux_c + tuy_c;
	double he5 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tuy_c - tux_c;
	double he6 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = -(tux_c + tuy_c);
	double he7 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));
	cidot3u = tux_c - tuy_c;
	double he8 = wdc*(omusq_c + cidot3u*(1.0 + 0.5*cidot3u));

	// calculate force population
	// temperory variables
	double coe0 = w0 / cs_square;
	double coes = ws / cs_square;
	double coed = wd / cs_square;

	double cflinv = 1.0 / CFL;

	double fpop0 = coe0*(-ux*forcex - uy*forcey);
	double fpop1 = coes*(((cflinv - ux) + (cflinv*ux)*cflinv / cs_square)*forcex - uy*forcey);
	double fpop2 = coes*(-ux*forcex + ((cflinv - uy) + (cflinv*uy)*cflinv / cs_square)*forcey);
	double fpop3 = coes*(((-cflinv - ux) + (cflinv*ux)*cflinv / cs_square)*forcex - uy*forcey);
	double fpop4 = coes*(-ux*forcex + ((-cflinv - uy) + (cflinv*uy)*cflinv / cs_square)*forcey);

	double cflinv2 = cflinv*cflinv / cs_square;
	double fpop5 = coed*(((cflinv - ux) + (ux + uy)*cflinv2)*forcex + ((cflinv - uy) + (ux + uy)*cflinv2)*forcey);
	double fpop6 = coed*(((-cflinv - ux) + (ux - uy)*cflinv2)*forcex + ((cflinv - uy) + (-ux + uy)*cflinv2)*forcey);
	double fpop7 = coed*(((-cflinv - ux) + (ux + uy)*cflinv2)*forcex + ((-cflinv - uy) + (ux + uy)*cflinv2)*forcey);
	double fpop8 = coed*(((cflinv - ux) + (ux - uy)*cflinv2)*forcex + ((-cflinv - uy) + (-ux + uy)*cflinv2)*forcey);

	// calculate f1 plus and minus
	double fp0 = ft0;
	// adjacent direction
	double fp1 = 0.5 * (ft1 + ft3);
	double fp2 = 0.5 * (ft2 + ft4);
	double fp3 = fp1;
	double fp4 = fp2;
	// diagonal direction
	double fp5 = 0.5 * (ft5 + ft7);
	double fp6 = 0.5 * (ft6 + ft8);
	double fp7 = fp5;
	double fp8 = fp6;

	double fm0 = 0.0;
	// adjacent direction
	double fm1 = 0.5 * (ft1 - ft3);
	double fm2 = 0.5 * (ft2 - ft4);
	double fm3 = -fm1;
	double fm4 = -fm2;
	// diagonal direction
	double fm5 = 0.5 * (ft5 - ft7);
	double fm6 = 0.5 * (ft6 - ft8);
	double fm7 = -fm5;
	double fm8 = -fm6;

	// calculate feq plus and minus
	double fep0 = fe0;
	// adjacent direction
	double fep1 = 0.5 * (fe1 + fe3);
	double fep2 = 0.5 * (fe2 + fe4);
	double fep3 = fep1;
	double fep4 = fep2;
	// diagonal direction
	double fep5 = 0.5 * (fe5 + fe7);
	double fep6 = 0.5 * (fe6 + fe8);
	double fep7 = fep5;
	double fep8 = fep6;

	double fem0 = 0.0;
	// adjacent direction
	double fem1 = 0.5 * (fe1 - fe3);
	double fem2 = 0.5 * (fe2 - fe4);
	double fem3 = -fem1;
	double fem4 = -fem2;
	// diagonal direction
	double fem5 = 0.5 * (fe5 - fe7);
	double fem6 = 0.5 * (fe6 - fe8);
	double fem7 = -fem5;
	double fem8 = -fem6;

	// calculate h1 plus and minus
	double hp0 = ht0;
	// adjacent direction
	double hp1 = 0.5 * (ht1 + ht3);
	double hp2 = 0.5 * (ht2 + ht4);
	double hp3 = hp1;
	double hp4 = hp2;
	// diagonal direction
	double hp5 = 0.5 * (ht5 + ht7);
	double hp6 = 0.5 * (ht6 + ht8);
	double hp7 = hp5;
	double hp8 = hp6;

	double hm0 = 0.0;
	// adjacent direction
	double hm1 = 0.5 * (ht1 - ht3);
	double hm2 = 0.5 * (ht2 - ht4);
	double hm3 = -hm1;
	double hm4 = -hm2;
	// diagonal direction
	double hm5 = 0.5 * (ht5 - ht7);
	double hm6 = 0.5 * (ht6 - ht8);
	double hm7 = -hm5;
	double hm8 = -hm6;

	// calculate heq plus and minus
	double hep0 = he0;
	// adjacent direction
	double hep1 = 0.5 * (he1 + he3);
	double hep2 = 0.5 * (he2 + he4);
	double hep3 = hep1;
	double hep4 = hep2;
	// diagonal direction
	double hep5 = 0.5 * (he5 + he7);
	double hep6 = 0.5 * (he6 + he8);
	double hep7 = hep5;
	double hep8 = hep6;

	double hem0 = 0.0;
	// adjacent direction
	double hem1 = 0.5 * (he1 - he3);
	double hem2 = 0.5 * (he2 - he4);
	double hem3 = -hem1;
	double hem4 = -hem2;
	// diagonal direction
	double hem5 = 0.5 * (he5 - he7);
	double hem6 = 0.5 * (he6 - he8);
	double hem7 = -hem5;
	double hem8 = -hem6;

	// calculate force_plus and force_minus
	double forcep0 = fpop0;
	double forcep1 = 0.5 * (fpop1 + fpop3);
	double forcep2 = 0.5 * (fpop2 + fpop4);
	double forcep3 = forcep1;
	double forcep4 = forcep2;
	double forcep5 = 0.5 * (fpop5 + fpop7);
	double forcep6 = 0.5 * (fpop6 + fpop8);
	double forcep7 = forcep5;
	double forcep8 = forcep6;

	double forcem0 = 0.0;
	double forcem1 = 0.5 * (fpop1 - fpop3);
	double forcem2 = 0.5 * (fpop2 - fpop4);
	double forcem3 = -forcem1;
	double forcem4 = -forcem2;
	double forcem5 = 0.5 * (fpop5 - fpop7);
	double forcem6 = 0.5 * (fpop6 - fpop8);
	double forcem7 = -forcem5;
	double forcem8 = -forcem6;

	double sp = 1.0 - 0.5*dt*omega_plus;
	double sm = 1.0 - 0.5*dt*omega_minus;

	double source0 = sp*fpop0;
	double source1 = sp*forcep1 + sm*forcem1;
	double source2 = sp*forcep2 + sm*forcem2;
	double source3 = sp*forcep3 + sm*forcem3;
	double source4 = sp*forcep4 + sm*forcem4;
	double source5 = sp*forcep5 + sm*forcem5;
	double source6 = sp*forcep6 + sm*forcem6;
	double source7 = sp*forcep7 + sm*forcem7;
	double source8 = sp*forcep8 + sm*forcem8;
	// ===============================================================
	//if (x == 5 && y == 1) {
	//	printf("%2.16g\n", charge);

	//printf("%g\n", source1);

	//}
	// ===============================================================
	// temporary variables (relaxation times)
	double tw0rp = omega_plus*dt;  //   omega_plus*dt 
	double tw0rm = omega_minus*dt; //   omega_minus*dt 
	double tw0cp = omega_c_plus*dt;  //   omega_c_plus*dt 
	double tw0cm = omega_c_minus*dt; //   omega_c_minus*dt 

	// TRT collision operations
	
	f0[gpu_field0_index(x, y)] = ft0 - (tw0rp * (fp0 - fep0) + tw0rm * (fm0 - fem0)) + dt*source0;
	h0[gpu_field0_index(x, y)] = ht0 - (tw0cp * (hp0 - hep0) + tw0cm * (hm0 - hem0));


	f2[gpu_fieldn_index(x, y, 1)] = ft1 - (tw0rp * (fp1 - fep1) + tw0rm * (fm1 - fem1)) + dt*source1;
	h2[gpu_fieldn_index(x, y, 1)] = ht1 - (tw0cp * (hp1 - hep1) + tw0cm * (hm1 - hem1));
	f2[gpu_fieldn_index(x, y, 2)] = ft2 - (tw0rp * (fp2 - fep2) + tw0rm * (fm2 - fem2)) + dt*source2;
	h2[gpu_fieldn_index(x, y, 2)] = ht2 - (tw0cp * (hp2 - hep2) + tw0cm * (hm2 - hem2));
	f2[gpu_fieldn_index(x, y, 3)] = ft3 - (tw0rp * (fp3 - fep3) + tw0rm * (fm3 - fem3)) + dt*source3;
	h2[gpu_fieldn_index(x, y, 3)] = ht3 - (tw0cp * (hp3 - hep3) + tw0cm * (hm3 - hem3));
	f2[gpu_fieldn_index(x, y, 4)] = ft4 - (tw0rp * (fp4 - fep4) + tw0rm * (fm4 - fem4)) + dt*source4;
	h2[gpu_fieldn_index(x, y, 4)] = ht4 - (tw0cp * (hp4 - hep4) + tw0cm * (hm4 - hem4));
	f2[gpu_fieldn_index(x, y, 5)] = ft5 - (tw0rp * (fp5 - fep5) + tw0rm * (fm5 - fem5)) + dt*source5;
	h2[gpu_fieldn_index(x, y, 5)] = ht5 - (tw0cp * (hp5 - hep5) + tw0cm * (hm5 - hem5));
	f2[gpu_fieldn_index(x, y, 6)] = ft6 - (tw0rp * (fp6 - fep6) + tw0rm * (fm6 - fem6)) + dt*source6;
	h2[gpu_fieldn_index(x, y, 6)] = ht6 - (tw0cp * (hp6 - hep6) + tw0cm * (hm6 - hem6));
	f2[gpu_fieldn_index(x, y, 7)] = ft7 - (tw0rp * (fp7 - fep7) + tw0rm * (fm7 - fem7)) + dt*source7;
	h2[gpu_fieldn_index(x, y, 7)] = ht7 - (tw0cp * (hp7 - hep7) + tw0cm * (hm7 - hem7));
	f2[gpu_fieldn_index(x, y, 8)] = ft8 - (tw0rp * (fp8 - fep8) + tw0rm * (fm8 - fem8)) + dt*source8;
	h2[gpu_fieldn_index(x, y, 8)] = ht8 - (tw0cp * (hp8 - hep8) + tw0cm * (hm8 - hem8));	
}

__global__ void gpu_boundary(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2, double *f0bc)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y;

	// set perturb = 0
	perturb = 0;

	// Boundary conditions
	double multis = 2.0*rho0*uw / cs_square * ws / CFL;
	double multid = 2.0*rho0*uw / cs_square * wd / CFL;

	// Full way bounce back
	if (y == 0) {
		// lower plate
		f0[gpu_field0_index(x, 0)]    = f0bc[gpu_field0_index(x, 0)];
		f2[gpu_fieldn_index(x, 0, 3)] = f1[gpu_fieldn_index(x, 0, 1)];
		f2[gpu_fieldn_index(x, 0, 4)] = f1[gpu_fieldn_index(x, 0, 2)];
		f2[gpu_fieldn_index(x, 0, 1)] = f1[gpu_fieldn_index(x, 0, 3)];
		f2[gpu_fieldn_index(x, 0, 2)] = f1[gpu_fieldn_index(x, 0, 4)];
		f2[gpu_fieldn_index(x, 0, 7)] = f1[gpu_fieldn_index(x, 0, 5)];
		f2[gpu_fieldn_index(x, 0, 8)] = f1[gpu_fieldn_index(x, 0, 6)];
		f2[gpu_fieldn_index(x, 0, 5)] = f1[gpu_fieldn_index(x, 0, 7)];
		f2[gpu_fieldn_index(x, 0, 6)] = f1[gpu_fieldn_index(x, 0, 8)];
		//if (x == 1) printf("%1.16g\n", f2[gpu_fieldn_index(x, 0, 2)]);
		return;
	}

	// direction numbering scheme
	// 6 2 5
	// 3 0 1
	// 7 4 8
	
	if (y ==  NY - 1) {
		// upper plate
		f0[gpu_field0_index(x, NY - 1)]    = f0bc[gpu_field0_index(x, 1)];
		f2[gpu_fieldn_index(x, NY - 1, 3)] = f1[gpu_fieldn_index(x, NY - 1, 1)] -multis;
		f2[gpu_fieldn_index(x, NY - 1, 4)] = f1[gpu_fieldn_index(x, NY - 1, 2)];
		f2[gpu_fieldn_index(x, NY - 1, 1)] = f1[gpu_fieldn_index(x, NY - 1, 3)] + multis;
		f2[gpu_fieldn_index(x, NY - 1, 2)] = f1[gpu_fieldn_index(x, NY - 1, 4)];
		f2[gpu_fieldn_index(x, NY - 1, 7)] = f1[gpu_fieldn_index(x, NY - 1, 5)] - multid;
		f2[gpu_fieldn_index(x, NY - 1, 8)] = f1[gpu_fieldn_index(x, NY - 1, 6)] + multid;
		f2[gpu_fieldn_index(x, NY - 1, 5)] = f1[gpu_fieldn_index(x, NY - 1, 7)] + multid;
		f2[gpu_fieldn_index(x, NY - 1, 6)] = f1[gpu_fieldn_index(x, NY - 1, 8)] - multid;

		// Zero charge gradient on Ny
		h0[gpu_field0_index(x, NY - 1)]    = h0[gpu_field0_index(x, NY - 2)];
		h2[gpu_fieldn_index(x, NY - 1, 1)] = h2[gpu_fieldn_index(x, NY - 2, 1)];
		h2[gpu_fieldn_index(x, NY - 1, 2)] = h2[gpu_fieldn_index(x, NY - 2, 2)];
		h2[gpu_fieldn_index(x, NY - 1, 3)] = h2[gpu_fieldn_index(x, NY - 2, 3)];
		h2[gpu_fieldn_index(x, NY - 1, 4)] = h2[gpu_fieldn_index(x, NY - 2, 4)];
		h2[gpu_fieldn_index(x, NY - 1, 5)] = h2[gpu_fieldn_index(x, NY - 2, 5)];
		h2[gpu_fieldn_index(x, NY - 1, 6)] = h2[gpu_fieldn_index(x, NY - 2, 6)];
		h2[gpu_fieldn_index(x, NY - 1, 7)] = h2[gpu_fieldn_index(x, NY - 2, 7)];
		h2[gpu_fieldn_index(x, NY - 1, 8)] = h2[gpu_fieldn_index(x, NY - 2, 8)];
		return;
	}
}

__global__ void gpu_stream(double *f0, double *f1, double *f2, double *h0, double *h1, double *h2)
{
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

	// streaming step

	unsigned int xp1 = (x + 1) % NX;
	unsigned int yp1 = (y + 1) % NY;
	unsigned int xm1 = (NX + x - 1) % NX;
	unsigned int ym1 = (NY + y - 1) % NY;

	// direction numbering scheme
	// 6 2 5
	// 3 0 1
	// 7 4 8

	// load populations from adjacent nodes (ft is post-streaming population of f1)
	f1[gpu_fieldn_index(x, y, 1)] = f2[gpu_fieldn_index(xm1, y, 1)];
	f1[gpu_fieldn_index(x, y, 2)] = f2[gpu_fieldn_index(x, ym1, 2)];
	f1[gpu_fieldn_index(x, y, 3)] = f2[gpu_fieldn_index(xp1, y, 3)];
	f1[gpu_fieldn_index(x, y, 4)] = f2[gpu_fieldn_index(x, yp1, 4)];
	f1[gpu_fieldn_index(x, y, 5)] = f2[gpu_fieldn_index(xm1, ym1, 5)];
	f1[gpu_fieldn_index(x, y, 6)] = f2[gpu_fieldn_index(xp1, ym1, 6)];
	f1[gpu_fieldn_index(x, y, 7)] = f2[gpu_fieldn_index(xp1, yp1, 7)];
	f1[gpu_fieldn_index(x, y, 8)] = f2[gpu_fieldn_index(xm1, yp1, 8)];

	h1[gpu_fieldn_index(x, y, 1)] = h2[gpu_fieldn_index(xm1, y, 1)];
	h1[gpu_fieldn_index(x, y, 2)] = h2[gpu_fieldn_index(x, ym1, 2)];
	h1[gpu_fieldn_index(x, y, 3)] = h2[gpu_fieldn_index(xp1, y, 3)];
	h1[gpu_fieldn_index(x, y, 4)] = h2[gpu_fieldn_index(x, yp1, 4)];
	h1[gpu_fieldn_index(x, y, 5)] = h2[gpu_fieldn_index(xm1, ym1, 5)];
	h1[gpu_fieldn_index(x, y, 6)] = h2[gpu_fieldn_index(xp1, ym1, 6)];
	h1[gpu_fieldn_index(x, y, 7)] = h2[gpu_fieldn_index(xp1, yp1, 7)];
	h1[gpu_fieldn_index(x, y, 8)] = h2[gpu_fieldn_index(xm1, yp1, 8)];
}

__global__ void gpu_bc_charge(double *h0, double *h1, double *h2)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y;

	if (y == 0) {
		double multi0c = 2.0*charge0*w0;
		double multisc = 2.0*charge0*ws;
		double multidc = 2.0*charge0*wd;
		// lower plate for charge density

		double ht1 = h2[gpu_fieldn_index(x, 0, 1)];
		double ht2 = h2[gpu_fieldn_index(x, 0, 2)];
		double ht3 = h2[gpu_fieldn_index(x, 0, 3)];
		double ht4 = h2[gpu_fieldn_index(x, 0, 4)];
		double ht5 = h2[gpu_fieldn_index(x, 0, 5)];
		double ht6 = h2[gpu_fieldn_index(x, 0, 6)];
		double ht7 = h2[gpu_fieldn_index(x, 0, 7)];
		double ht8 = h2[gpu_fieldn_index(x, 0, 8)];
		// lower plate for constant charge density

		h0[gpu_field0_index(x, 0)] = -h0[gpu_field0_index(x, 0)] + multi0c;
		h1[gpu_fieldn_index(x, 0, 3)] = -ht1 + multisc;
		h1[gpu_fieldn_index(x, 0, 4)] = -ht2 + multisc;
		h1[gpu_fieldn_index(x, 0, 1)] = -ht3 + multisc;
		h1[gpu_fieldn_index(x, 0, 2)] = -ht4 + multisc;
		h1[gpu_fieldn_index(x, 0, 7)] = -ht5 + multidc;
		h1[gpu_fieldn_index(x, 0, 8)] = -ht6 + multidc;
		h1[gpu_fieldn_index(x, 0, 5)] = -ht7 + multidc;
		h1[gpu_fieldn_index(x, 0, 6)] = -ht8 + multidc;
	}
}


__host__ void compute_parameters(double *T, double *M, double *C, double *Fe) {
	double K_host;
	double eps_host;
	double voltage_host;
	//double nu_host;
	double Ly_host;
	double diffu_host;
	double charge0_host;
	double rho0_host;

	cudaMemcpyFromSymbol(&K_host, K, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&eps_host, eps, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&voltage_host, voltage, sizeof(double), 0, cudaMemcpyDeviceToHost);
	//cudaMemcpyFromSymbol(&nu_host, nu, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&Ly_host, Ly, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&diffu_host, diffu, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&charge0_host, charge0, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&rho0_host, rho0, sizeof(double), 0, cudaMemcpyDeviceToHost);

	*M = sqrt(eps_host / rho0_host) / K_host;
	*T = eps_host*voltage_host / K_host / nu_host / rho0_host;
	*C = charge0_host * Ly_host * Ly_host / (voltage_host * eps_host);
	*Fe = K_host * voltage_host / diffu_host;
}

__host__ void report_flow_properties(unsigned int n, double t, double *rho, 
	double *charge, double *phi, double *ux, double *uy, double *Ex, double *Ey)
{
    printf("Iteration: %u, physical time: %g.\n",n,t);
}

__host__ void save_scalar(const char* name, double *scalar_gpu, double *scalar_host, unsigned int n)
{
    // assume reasonably-sized file names
    char filename[128];
    char format[16];
    
    // compute maximum number of digits
    int ndigits = floor(log10((double)NSTEPS)+1.0);
    
    // generate format string
    // file name format is name0000nnn.bin
    sprintf(format,"%%s%%0%dd.bin",ndigits);
    sprintf(filename,format,name,n);
    
    // transfer memory from GPU to host
    checkCudaErrors(cudaMemcpy(scalar_host,scalar_gpu,mem_size_scalar,cudaMemcpyDeviceToHost));
    
    // open file for writing
    FILE *fout = fopen(filename,"wb+");
    
    // write data
    fwrite(scalar_host,1,mem_size_scalar,fout);
    
    // close file
    fclose(fout);
    
    if(ferror(fout))
    {
        fprintf(stderr,"Error saving to %s\n",filename);
        perror("");
    }
    else
    {
        if(!quiet)
            printf("Saved to %s\n",filename);
    }
}

__host__
void save_data_tecplot(double time, double *rho_gpu, double *charge_gpu, double *phi_gpu,
	double *ux_gpu, double *uy_gpu, double *Ex_gpu, double *Ey_gpu) {
	
	double *rho    = (double*)malloc(mem_size_scalar);
	double *charge = (double*)malloc(mem_size_scalar);
	double *phi    = (double*)malloc(mem_size_scalar);
	double *ux     = (double*)malloc(mem_size_scalar);
	double *uy     = (double*)malloc(mem_size_scalar);
	double *Ex     = (double*)malloc(mem_size_scalar);
	double *Ey     = (double*)malloc(mem_size_scalar);
	double dx_host;
	double dy_host;
	// transfer memory from GPU to host
	checkCudaErrors(cudaMemcpy(rho,    rho_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(charge, charge_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(phi,    phi_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(ux,     ux_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(uy,     uy_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ex,     Ex_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(Ey,     Ey_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	cudaMemcpyFromSymbol(&dx_host, dx, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&dy_host, dy, sizeof(double), 0, cudaMemcpyDeviceToHost);
	
	
	// apply boundary conditions (upper and lower plate)
	for (unsigned int x = 0; x < NX; ++x) {
		rho[scalar_index(x, 0)]         = 2.0*rho[scalar_index(x, 1)] - rho[scalar_index(x, 2)];
		charge[scalar_index(x, 0)]      = 2.0*charge[scalar_index(x, 1)] - charge[scalar_index(x, 2)];
		ux[scalar_index(x, 0)]          = 2.0*ux[scalar_index(x, 1)] - ux[scalar_index(x, 2)];
		uy[scalar_index(x, 0)]          = 2.0*uy[scalar_index(x, 1)] - uy[scalar_index(x, 2)];
		rho[scalar_index(x, NY - 1)]    = 2.0*rho[scalar_index(x, NY - 2)] - rho[scalar_index(x, NY - 3)];
		charge[scalar_index(x, NY - 1)] = 2.0*charge[scalar_index(x, NY - 2)] - charge[scalar_index(x, NY - 3)];
		ux[scalar_index(x, NY - 1)]     = 2.0*ux[scalar_index(x, NY - 2)] - ux[scalar_index(x, NY - 3)];
		uy[scalar_index(x, NY - 1)]     = 2.0*uy[scalar_index(x, NY - 2)] - uy[scalar_index(x, NY - 3)];
	}


	// open file for writing
	FILE *fout = fopen("data.dat", "wb+");
	char str[] = "VARIABLES=\"x\",\"y\",\"u\",\"v\",\"p\",\"charge\",\"phi\",\"Ex\",\"Ey\"";
	//fwrite(str, 1, sizeof(str), fout);
	fprintf(fout, "%s\n", str);
	fprintf(fout, "ZONE T=\"t=%g\", F=POINT, I = %d, J = %d\n", time, NX, NY);

	for (unsigned int y = 0; y < NY; ++y)
	{
		for (unsigned int x = 0; x < NX; ++x)
		{
			//double data[] = { dx*x, dy*y, u[scalar_index(x, y)], v[scalar_index(x, y)], r[scalar_index(x, y)], c[scalar_index(x, y)], fi[scalar_index(x, y)], ex[scalar_index(x, y)], ey[scalar_index(x, y)] };
			fprintf(fout, "%g %g %g %g %g %g %10.6f %10.6f %10.6f\n", dx_host*x, dy_host*y,
				ux[scalar_index(x, y)], uy[scalar_index(x, y)], rho[scalar_index(x, y)], charge[scalar_index(x, y)], 
				phi[scalar_index(x, y)], Ex[scalar_index(x, y)], Ey[scalar_index(x, y)]);
			//printf("X is %g and Y is %g\n", dx_host*x, dy_host*y);
		}
	}


	//printf("%g\n", u[scalar_index(10, 10)]);
	fclose(fout);
}

/*
__host__
void poisson_phi(double *charge_gpu, double *phi_gpu)
{
	double voltage_host;
	double *c = (double*)malloc(mem_size_scalar);
	double *fi = (double*)malloc(mem_size_scalar);
	cudaMemcpyFromSymbol(&voltage_host, voltage, sizeof(double), 0, cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaMemcpy(c, charge_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(fi, phi_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	// Iteration parameters
	int MAX_ITERATIONS = 3.0e6;
	double TOLERANCE = 1.0e-9;
	double w = 1.6;
	unsigned int it;
	double R;
	// Boundary conditions
	for (unsigned int x = 0; x < NX; ++x)
	{
		fi[scalar_index(x, 0)] = voltage_host;
		fi[scalar_index(x, NY - 1)] = 0.0;
	}
	for (it = 0; it<MAX_ITERATIONS; ++it)
	{
		R = 0.0;
		for (unsigned int y = 1; y < NY - 1; ++y) // loop around internal grid points, excluding the upper and lower boundaries
		{
			for (unsigned int x = 0; x < NX; ++x) // loop from 0 -> N-1
			{
				unsigned int xp1 = (x + 1) % NX;
				unsigned int yp1 = (y + 1) % NY;
				unsigned int xm1 = (NX + x - 1) % NX;
				unsigned int ym1 = (NY + y - 1) % NY;

				double charge = c[scalar_index(x, y)];
				double phi = fi[scalar_index(x, y)];
				double phiL = fi[scalar_index(xm1, y)];
				double phiR = fi[scalar_index(xp1, y)];
				double phiU = fi[scalar_index(x, yp1)];
				double phiD = fi[scalar_index(x, ym1)];
				double source = (charge / eps) * dx *dx; // Right hand side of the equation
				double phi_old = phi;
				phi = 0.25 * (phiL + phiR + phiU + phiD + source);
				phi = w*phi + (1 - w)*phi_old;
				if (R < fabs(phi - phi_old)) {
					R = fabs(phi - phi_old);
				}
				fi[scalar_index(x, y)] = phi;
			}
		}
		// break loop if convergence
		if (R < TOLERANCE) {
			break;
		}

		// report residual
		//if (it % 10 == 1 && test == 1) {
		//	printf("Residual = %g\n", R);
		//}
	}
	if (it == MAX_ITERATIONS) {
		printf("Poisson solver did not converge!\n");
		printf("Residual = %g\n", R);
		system("pause");
		exit(-1);
	}
	checkCudaErrors(cudaMemcpy(phi_gpu, fi, mem_size_scalar, cudaMemcpyHostToDevice));
}

__host__
void efield(double *phi_gpu, double *Ex_gpu, double *Ey_gpu) {
	double *Ex = (double*)malloc(mem_size_scalar);
	double *Ey = (double*)malloc(mem_size_scalar);
	double *fi = (double*)malloc(mem_size_scalar);
	checkCudaErrors(cudaMemcpy(fi, phi_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	for (unsigned int y = 1; y < NY - 1; ++y) // loop around internal grid points, excluding the upper and lower boundaries
	{
		for (unsigned int x = 0; x < NX; ++x) // loop from 0 -> N-1
		{
			unsigned int xp1 = (x + 1) % NX;
			unsigned int yp1 = (y + 1) % NY;
			unsigned int xm1 = (NX + x - 1) % NX;
			unsigned int ym1 = (NY + y - 1) % NY;

			double phi = fi[scalar_index(x, y)];
			double phiL = fi[scalar_index(xm1, y)];
			double phiR = fi[scalar_index(xp1, y)];
			double phiU = fi[scalar_index(x, yp1)];
			double phiD = fi[scalar_index(x, ym1)];

			Ex[scalar_index(x, y)] = 0.5*(phiL - phiR) / dx;
			Ey[scalar_index(x, y)] = 0.5*(phiD - phiU) / dy;

		}
	}
	for (unsigned int x = 0; x < NX; ++x) {
		//ex[scalar_index(x, 0)] = ex[scalar_index(x, 1)];
		Ey[scalar_index(x, 0)] = Ey[scalar_index(x, 1)];
		//ex[scalar_index(x, NY-1)] = ex[scalar_index(x, NY-2)];
		Ey[scalar_index(x, NY - 1)] = Ey[scalar_index(x, NY - 2)];
	}
	checkCudaErrors(cudaMemcpy(Ex_gpu, Ex, mem_size_scalar, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Ey_gpu, Ey, mem_size_scalar, cudaMemcpyHostToDevice));

}*/
