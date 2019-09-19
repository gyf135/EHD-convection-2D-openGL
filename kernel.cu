/* This code accompanies
 *   Two relaxation time lattice Boltzmann method coupled to fast Fourier transform Poisson solver: Application to electroconvective flow, Journal of Computational Physics
 *	 https://doi.org/10.1016/j.jcp.2019.07.029
 *   Numerical analysis of electroconvection in cross-flow with unipolar charge injection, Physical Review Fluids
 *	 
 *   Yifei Guan, Igor Novosselov
 * 	 University of Washington
 *
 * Author: Yifei Guan
 *
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include "LBM.h"
#include <device_functions.h>

__device__
unsigned char clip(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__global__ void charge_visualize(uchar4*, double*);

__host__
void kernelLauncher() {

	// stream and collide from f1 storing to f2
	// optionally compute and save moments
	stream_collide_save(f0_gpu, f1_gpu, f2_gpu, h0_gpu, h1_gpu, h2_gpu, rho_gpu, charge_gpu,
		ux_gpu, uy_gpu, Ex_gpu, Ey_gpu, t, f0bc);
	// =========================================================================
	// Fast poisson solver
	// =========================================================================

	// Extend the domain
	extension(charge_gpu, charge_gpu_ext);

	// Execute a real-to-complex 2D FFT
	CHECK_CUFFT(cufftExecZ2Z(plan, charge_gpu_ext, freq_gpu_ext, CUFFT_FORWARD));

	// Execute the derivatives in frequency domain
	derivative(kx, ky, freq_gpu_ext);

	// Execute a complex-to-complex 2D IFFT
	CHECK_CUFFT(cufftExecZ2Z(plan, freq_gpu_ext, phi_gpu_ext, CUFFT_INVERSE));

	// Extraction of phi from extended domain phi_gpu_ext
	extract(phi_gpu, phi_gpu_ext);

	// Calculate electric field strength
	efield(phi_gpu, Ex_gpu, Ey_gpu);

	t = t + dt_host;
}

__global__ void charge_visualize(uchar4 *d_out, double *charge) {
	unsigned int y = blockIdx.y;
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if (y==NY-1) {
		charge[gpu_scalar_index(x, NY - 1)] = 2.0*charge[gpu_scalar_index(x, NY - 2)] - charge[gpu_scalar_index(x, NY - 3)];
	}
	if (y == 0) {
		charge[gpu_scalar_index(x, 0)] = 2.0*charge[gpu_scalar_index(x, 1)] - charge[gpu_scalar_index(x, 2)];
	}
	double a = (1 - charge[gpu_scalar_index(x, y)] / 2) / 0.2;
	if (a <= 0) {
		d_out[gpu_scalar_index(x, NY - 1 - y)].x = 255; // Red
		d_out[gpu_scalar_index(x, NY - 1 - y)].y = 0; // Green
		d_out[gpu_scalar_index(x, NY - 1 - y)].z = 0; // Blue
		return;
	}
	int i = floorf(a);
	int j = floorf(255 * (a - i));
	switch (i) {
	case 0: 
		d_out[gpu_scalar_index(x, NY - 1 - y)].x = 255; // Red
		d_out[gpu_scalar_index(x, NY - 1 - y)].y = j; // Green
		d_out[gpu_scalar_index(x, NY - 1 - y)].z = 0; // Blue
		break;
	case 1:
		d_out[gpu_scalar_index(x, NY - 1 - y)].x = 255-j; // Red
		d_out[gpu_scalar_index(x, NY - 1 - y)].y = 255; // Green
		d_out[gpu_scalar_index(x, NY - 1 - y)].z = 0; // Blue
		break;
	case 2:
		d_out[gpu_scalar_index(x, NY - 1 - y)].x = 0; // Red
		d_out[gpu_scalar_index(x, NY - 1 - y)].y = 255; // Green
		d_out[gpu_scalar_index(x, NY - 1 - y)].z = j; // Blue
		break;
	case 3:
		d_out[gpu_scalar_index(x, NY - 1 - y)].x = 0; // Red
		d_out[gpu_scalar_index(x, NY - 1 - y)].y = 255-j; // Green
		d_out[gpu_scalar_index(x, NY - 1 - y)].z = 255; // Blue
		break;
	case 4:
		d_out[gpu_scalar_index(x, NY - 1 - y)].x = j; // Red
		d_out[gpu_scalar_index(x, NY - 1 - y)].y = 0; // Green
		d_out[gpu_scalar_index(x, NY - 1 - y)].z = 255; // Blue
	case 5:
		d_out[gpu_scalar_index(x, NY - 1 - y)].x = 0; // Red
		d_out[gpu_scalar_index(x, NY - 1 - y)].y = 0; // Green
		d_out[gpu_scalar_index(x, NY - 1 - y)].z = 255; // Blue
	}




	//const unsigned char intensity = clip((int)(charge[gpu_scalar_index(x, y)]/5*255));
	//d_out[gpu_scalar_index(x, NY - 1 - y)].x = intensity; // higher charge -> more red
	//d_out[gpu_scalar_index(x, NY - 1 - y)].y = 255-intensity; // lower charge -> more green
}
/*
__host__ void perturbFlow() {
	// blocks in grid
	dim3  grid(NX / nThreads, NY, 1);
	// threads in block
	dim3  threads(nThreads, 1, 1);

	gpu_perturbFlow << < grid, threads >> >(ux_gpu, uy_gpu);
	getLastCudaError("gpu_perturbFlow kernel error");
}*/

__host__ double current(double *c, double *ey) {
	double I = 0;
	for (unsigned int x = 0; x < NX; x++) {
		I += c[scalar_index(x, NY - 1)] * ey[scalar_index(x, NY - 1)];
	}
	I = I * K_host *dy_host;
	return I;
}


