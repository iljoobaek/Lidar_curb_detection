#include<stdio.h>
#include <pthread.h>
#define NUM_THREAD =4

void* printThread(void *pArg) {
	int*p = (int*)pArg;
	int myNum = *p;
		printf("Thread num %d\n", myNum);
	return 0;
}

int main(void) {
	pthread_t tid[NUM_THREAD];
	for (int i = 0; i < NUM_THREAD; i++) {
		pthread_create(&tid[i], NULL, printThread, &i);
	}
	for (int j = 0; j < NUM_THREAD; j++) {
		pthread_join(tid[i], NULL);
	}
	return 0;
}