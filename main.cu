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
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#include "seconds.h"
#include "LBM.h"
#include "LBM.cu"
#include "kernel.cu"
#include "poisson.cu"
#include <cuda_runtime.h>
#include <cufft.h>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <gl/glew.h>
#include <gl/freeglut.h>
#endif
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "interactions.h"
#include "main_2.cu"



int main(int argc, char* argv[])
{
	cudaMemcpyFromSymbol(&dt_host, dt, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&Lx_host, Lx, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&Ly_host, Ly, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&dy_host, dy, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cudaMemcpyFromSymbol(&K_host, K, sizeof(double), 0, cudaMemcpyDeviceToHost);

	cudaMemcpyToSymbol(nu, &nu_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(uw, &uw_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(exf, &exf_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(charge0, &charge0_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(voltage, &voltage_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	// Compute parameters
	compute_parameters(T, M, C, Fe);

    printf("Simulating 2D electroconvection\n");
    printf("By: Yifei Guan, University of Washington\n");
    printf("      domain size: %ux%u\n",NX,NY);
    printf("                T: %g\n",*T);
    printf("                M: %g\n",*M);
    printf("                C: %g\n",*C);
    printf("               Fe: %g\n",*Fe);
    printf("        timesteps: %u\n",NSTEPS);
    printf("       save every: %u\n",NSAVE);
    printf("    message every: %u\n",NMSG);
    printf("\n");
    
    double bytesPerMiB = 1024.0*1024.0;
    double bytesPerGiB = 1024.0*1024.0*1024.0;
    
    checkCudaErrors(cudaSetDevice(0));
    int deviceId = 0;
    checkCudaErrors(cudaGetDevice(&deviceId));
    
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, deviceId));
    
    size_t gpu_free_mem, gpu_total_mem;
    checkCudaErrors(cudaMemGetInfo(&gpu_free_mem,&gpu_total_mem));
    
    printf("CUDA information\n");
    printf("       using device: %d\n", deviceId);
    printf("               name: %s\n",deviceProp.name);
    printf("    multiprocessors: %d\n",deviceProp.multiProcessorCount);
    printf(" compute capability: %d.%d\n",deviceProp.major,deviceProp.minor);
    printf("      global memory: %.1f MiB\n",deviceProp.totalGlobalMem/bytesPerMiB);
    printf("        free memory: %.1f MiB\n",gpu_free_mem/bytesPerMiB);
    printf("\n");

	// print instructions
	printf("Interaction instructions\n");
	printf("Press \'a\'  to purturb\n");
	printf("Press \'->\' to increase uwall\n");
	printf("Press \'<-\' to decrease uwall\n");
	printf("Press \'w\'  to increase external force\n");
	printf("Press \'s\'  to decrease external force\n");
	printf("Press \'e\'  to decrease applied voltage\n");
	printf("Press \'r\'  to increase applied voltage\n");
	printf("Press \'d\'  to decrease applied charge density\n");
	printf("Press \'f\'  to increase applied charge density\n");
	printf("Press \'j\'  to decrease viscosity of fluid\n");
	printf("Press \'k\'  to increase viscosity of fluid\n");
	// storage of f0 at upper and lower plate
	checkCudaErrors(cudaMalloc((void**)&f0bc, sizeof(double)*NX * 2));
    //double *prop_gpu;
	// microscopic variables
	checkCudaErrors(cudaMalloc((void**)&f0_gpu, mem_size_0dir));
	checkCudaErrors(cudaMalloc((void**)&f1_gpu, mem_size_n0dir));
	checkCudaErrors(cudaMalloc((void**)&f2_gpu, mem_size_n0dir));
	checkCudaErrors(cudaMalloc((void**)&h0_gpu, mem_size_0dir));
	checkCudaErrors(cudaMalloc((void**)&h1_gpu, mem_size_n0dir));
	checkCudaErrors(cudaMalloc((void**)&h2_gpu, mem_size_n0dir));
	// macroscopic variables
	checkCudaErrors(cudaMalloc((void**)&rho_gpu,    mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&ux_gpu,     mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&uy_gpu,     mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&charge_gpu, mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&phi_gpu,    mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&Ex_gpu,     mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&Ey_gpu,     mem_size_scalar));
	checkCudaErrors(cudaMalloc((void**)&kx,         sizeof(double)*NX));
	checkCudaErrors(cudaMalloc((void**)&ky,         sizeof(double)*NE));

    // Setup the cuFFT plan
	CHECK_CUFFT(cufftPlan2d(&plan, NE, NX, CUFFT_Z2Z));
	checkCudaErrors(cudaMalloc((void**)&freq_gpu_ext, sizeof(cufftDoubleComplex)*NX*NE));
	checkCudaErrors(cudaMalloc((void**)&phi_gpu_ext,  sizeof(cufftDoubleComplex)*NX*NE));
	checkCudaErrors(cudaMalloc((void**)&charge_gpu_ext, sizeof(cufftDoubleComplex)*NX*NE));


	// Setup the frequencies kx and ky
	for (unsigned i = 0; i <= NX / 2; i++)
	{
		kx_host[i] = (double)i * 2.0 * M_PI / Lx_host;
	}

	for (unsigned i = NX / 2 + 1; i < NX; i++)
	{
		kx_host[i] = ((double) i - NX) * 2.0 * M_PI / Lx_host;
	}
	for (unsigned i = 0; i <= NE / 2; i++)
	{
		ky_host[i] = (double)i  * 2.0 * M_PI / (NE*dy_host);
	}

	for (unsigned i = NE / 2 + 1; i < NE; i++)
	{
		ky_host[i] = ((double)i - NE) * 2.0 * M_PI / (NE*dy_host);
	}

	CHECK(cudaMemcpy(kx, kx_host,
		sizeof(double) * NX, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ky, ky_host,
		sizeof(double) * NE, cudaMemcpyHostToDevice));
	
    // create event objects
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

	// Zero flow at t=0
	// to initialize rho, charge, phi, ux, uy, Ex, Ey fields.
	initialization(rho_gpu, charge_gpu, phi_gpu, ux_gpu, uy_gpu, Ex_gpu, Ey_gpu);
       
    // initialise f1,h1 as equilibrium for rho, ux, uy, charge, ex, ey
    init_equilibrium(f0_gpu,f1_gpu,h0_gpu,h1_gpu, rho_gpu,charge_gpu, ux_gpu,uy_gpu,Ex_gpu,Ey_gpu);

	// report computational results to screen
    //report_flow_properties(0, t, rho_gpu, charge_gpu, phi_gpu, ux_gpu,uy_gpu, Ex_gpu, Ey_gpu);
    
    double begin = seconds();
    checkCudaErrors(cudaEventRecord(start,0));

    // main simulation loop; take NSTEPS time steps
	// Visualizing the flow field
	initGLUT(&argc, argv);
	gluOrtho2D(0, NX, NY, 0);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(handleSpecialKeypress);
	glutIdleFunc(idle);
	glutDisplayFunc(display);
	initPixelBuffer();
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
		GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutMainLoop();
	//atexit(exitfunc);
	//perturbFlow();
    checkCudaErrors(cudaEventRecord(stop,0));
    checkCudaErrors(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds,start,stop));
    
    double end = seconds();
    double runtime = end-begin;
    double gpu_runtime = 0.001*milliseconds;	
	
	size_t doubles_read = ndir; // per node every time step
    size_t doubles_written = ndir;
    size_t doubles_saved = 3; // per node every NSAVE time steps
    
    // note NX*NY overflows when NX=NY=65536
    size_t nodes_updated = NSTEPS*size_t(NX*NY);
    size_t nodes_saved   = (NSTEPS/NSAVE)*size_t(NX*NY);
    double speed = nodes_updated/(1e6*runtime);
    
    double bandwidth = (nodes_updated*(doubles_read + doubles_written)+nodes_saved*(doubles_saved))*sizeof(double)/(runtime*bytesPerGiB);
    
    printf(" ----- performance information -----\n");
    printf("               timesteps: %u\n",NSTEPS);
    printf("           clock runtime: %.3f (s)\n",runtime);
    printf("             gpu runtime: %.3f (s)\n",gpu_runtime);
    printf("                   speed: %.2f (Mlups)\n",speed);
    
	save_data_tecplot(t, rho_gpu, charge_gpu, phi_gpu, ux_gpu, uy_gpu, Ex_gpu, Ey_gpu);

    // destory event objects
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    
    // free all memory allocatd on the GPU and host
    checkCudaErrors(cudaFree(f0_gpu));
    checkCudaErrors(cudaFree(f1_gpu));
    checkCudaErrors(cudaFree(f2_gpu));
	checkCudaErrors(cudaFree(h0_gpu));
	checkCudaErrors(cudaFree(h1_gpu));
	checkCudaErrors(cudaFree(h2_gpu));
    checkCudaErrors(cudaFree(rho_gpu));
	checkCudaErrors(cudaFree(phi_gpu));
	checkCudaErrors(cudaFree(Ex_gpu));
	checkCudaErrors(cudaFree(Ey_gpu));
    checkCudaErrors(cudaFree(ux_gpu));
    checkCudaErrors(cudaFree(uy_gpu));
	checkCudaErrors(cudaFree(f0bc));
	checkCudaErrors(cudaFree(charge_gpu_ext));
	checkCudaErrors(cudaFree(phi_gpu_ext));
	checkCudaErrors(cudaFree(freq_gpu_ext));
	checkCudaErrors(cudaFree(kx));
	checkCudaErrors(cudaFree(ky));
	CHECK_CUFFT(cufftDestroy(plan));
    //checkCudaErrors(cudaFree(prop_gpu));    
	free(kx_host);
	free(ky_host);
    
    // release resources associated with the GPU device
    cudaDeviceReset();
	system("pause");
    return 0;
}

