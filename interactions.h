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
#ifndef INTERACTIONS_H
#define INTERACTIONS_H
#define DELTA 0.1 //pixel increment for arrow keys
#define TITLE_STRING "flashlight: distance image display app"
bool dragMode = false; // mouse tracking mode

void keyboard(unsigned char key, int x, int y) {
	if (key == 'w') {
		exf_host += DELTA * 100;
		cudaMemcpyToSymbol(exf, &exf_host, sizeof(double), 0, cudaMemcpyHostToDevice);
		return;
	}
	if (key == 's') {
		exf_host -= DELTA * 100;
		cudaMemcpyToSymbol(exf, &exf_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}

	if (key == 'e') {
		voltage_host -= 100;
		cudaMemcpyToSymbol(voltage, &voltage_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}
	if (key == 'r') {
		voltage_host += 100;
		cudaMemcpyToSymbol(voltage, &voltage_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}
	if (key == 'd') {
		charge0_host -= 1;
		cudaMemcpyToSymbol(charge0, &charge0_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}
	if (key == 'f') {
		charge0_host += 1;
		cudaMemcpyToSymbol(charge0, &charge0_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}
	if (key == 'j') {
		nu_host -= 0.001;
		cudaMemcpyToSymbol(nu, &nu_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}
	if (key == 'k') {
		nu_host += 0.001;
		cudaMemcpyToSymbol(nu, &nu_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}
	if (key == 'a') {
		perturb_host = 1; // Flow perturbation
		cudaMemcpyToSymbol(perturb, &perturb_host, sizeof(int), 0, cudaMemcpyHostToDevice);
		printf("Applied perturbation\n");
	}
	if (key == '27') exit(0);	
	glutPostRedisplay();
}
/*
void mouseMove(int x, int y) {
	if (dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}

void mouseDrag(int x, int y) {
	if (!dragMode) return;
	loc.x = x;
	loc.y = y;
	glutPostRedisplay();
}
void printInstructions() {
printf("flashlight interactions\n");
printf("a:toggle mouse tracking mode\n");
printf("arrow keys: move ref location\n");
printf("esc: close graphics window\n");
}
*/


void handleSpecialKeypress(int key, int x, int y) {
	if (key == GLUT_KEY_LEFT) {
		uw_host -= DELTA*0.1;
		cudaMemcpyToSymbol(uw, &uw_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}
	if (key == GLUT_KEY_RIGHT) {
		uw_host += DELTA*0.1;
		cudaMemcpyToSymbol(uw, &uw_host, sizeof(double), 0, cudaMemcpyHostToDevice);
	}
	glutPostRedisplay();
}

void idle(void) {
	++iteractionCount;
	glutPostRedisplay();
}
#endif
