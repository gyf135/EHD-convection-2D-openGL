/* This code accompanies
 *   Two relaxation time lattice Boltzmann method coupled to fast Fourier transform Poisson solver: Application to electroconvective flow, Journal of Computational Physics
 *	 https://doi.org/10.1016/j.jcp.2019.07.029
 *	 Numerical analysis of electroconvection in cross-flow with unipolar charge injection, Physical Review Fluids
 *	 
 *   Yifei Guan, Igor Novosselov
 * 	 University of Washington
 *
 * Author: Yifei Guan
 *
 */
#include <stdio.h>
#include <stdlib.h>
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
#include "LBM.h"
#include <math.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h>
#include <device_functions.h>

// Texture and pixel objects
GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;

void render() {
	uchar4 *d_out = 0;
	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
	for (int i = 0; i < NSTEPS; i++) {
		kernelLauncher();
	}
	// blocks in grid
	dim3 grid(NX / nThreads, NY, 1);
	// threads in block
	dim3 threads(nThreads, 1, 1);
	charge_visualize << <grid, threads >> > (d_out, charge_gpu);
	getLastCudaError("Charge visualization error");

	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
	
	// Compute parameters
	compute_parameters(T, M, C, Fe);

	double up_host = 7.8125e-5*exf_host*Ly_host*Ly_host / nu_host;

	if (exf_host > -1 && exf_host < 1) {
		parameter = (*C)*(*T);
		X = (*C)*(*M)*(*M);
	}
	else {
		double shear = 1600*nu_host * up_host / Ly_host * 2.0;
		parameter = charge0_host * voltage_host / shear;
		X = (charge0_host*voltage_host) / (1600.0*up_host*up_host);
	}

	checkCudaErrors(cudaMemcpy(Ey_host, Ey_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(charge_host, charge_gpu, mem_size_scalar, cudaMemcpyDeviceToHost));
	double current_host;
	current_host = current(charge_host, Ey_host);

	char title[250];
	sprintf(title, "Iterations=%d, "
		"T = %g, C = %g, M = %g, Fe = %g, uwall = %g, external force = %g, CxT = %g, X= %g, current = %g",
		iteractionCount*NSTEPS, *T, *C, *M, *Fe, uw_host,exf_host, parameter,X,current_host);
	glutSetWindowTitle(title);
}

void drawTexture() {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, H);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(W, H);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(W, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

void display() {
	render();
	drawTexture();
	glutSwapBuffers();
}

void initGLUT(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(W, H);
	glutCreateWindow(TITLE_STRING);
#ifndef __APPLE__
	glewInit();
#endif
}

void initPixelBuffer() {
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * W*H * sizeof(GLubyte), 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
}
/*
int main(int argc, char** argv) {
printInstructions();
initGLUT(&argc, argv);
gluOrtho2D(0, W, H, 0);
glutKeyboardFunc(keyboard);
glutSpecialFunc(handleSpecialKeypress);
glutPassiveMotionFunc(mouseMove);
glutMotionFunc(mouseDrag);
glutDisplayFunc(display);
initPixelBuffer();
glutMainLoop();
atexit(exitfunc);
return 0;
}*/