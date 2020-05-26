#include <errno.h>
#include <fcntl.h>
#include <linux/limits.h>
#include <sys/mman.h> 
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include <sys/time.h>
#include <sys/syscall.h>
#include <sched.h>
#include <pthread.h>


#include "global_init_time.h"


//#define GLOBAL_INIT_TIME_DEBUG

#ifdef GLOBAL_INIT_TIME_DEBUG
	#define Global_Init_time_DPNT(fmt, args...)		fprintf(stdout, fmt, ## args)
	#define Global_Init_time_EPNT(fmt, args...)		fprintf(stderr, fmt, ## args)
#else
	#define Global_Init_time_DPNT(fmt, args...)
	#define Global_Init_time_EPNT(fmt, args...)		fprintf(stderr, fmt, ## args)
#endif


// Create or open shared memory and a global init_time
global_init_time_t global_init_time_init(char *name) {
    global_init_time_t init_timeHdlr = {NULL, 0, NULL, 0};
    errno = 0;

    // Open shared memory, or create one.
    // "created" flag indicates if the shared memory was intialized for the first time
    init_timeHdlr.shm_fd = shm_open(name, O_RDWR, 0660);
    if (errno == ENOENT) {
        init_timeHdlr.shm_fd = shm_open(name, O_RDWR|O_CREAT, 0660);
        init_timeHdlr.created = 1;
        Global_Init_time_DPNT("[Global__Init_time] global_init_time shared memory is created for the first time!!\n");
    }
    if (init_timeHdlr.shm_fd == -1) {
        perror("shm_open");
        return init_timeHdlr;
    }

    // Truncate global memory segment so it would contain
    if (ftruncate(init_timeHdlr.shm_fd, sizeof(struct timespec)) != 0) {
        perror("ftruncate");
        return init_timeHdlr;
    }

    // Map pthread mutex into the global memory.
    void *addr = mmap(
                        NULL,
                        sizeof(struct timespec),
                        PROT_READ|PROT_WRITE,
                        MAP_SHARED,
                        init_timeHdlr.shm_fd,
                        0
                        );
    if (addr == MAP_FAILED) {
        perror("mmap");
        return init_timeHdlr;
    }
    struct timespec *init_time_ptr = (struct timespec *)addr;

    // If global memory was just initialized
    // initialize the mutex as well.
    if (init_timeHdlr.created) {
        clock_gettime(CLOCK_REALTIME, init_time_ptr);
        Global_Init_time_DPNT("[Global__Init_time] set initial time : %llu\n", init_time_ptr->tv_sec * 1000000000ULL + init_time_ptr->tv_nsec);
    }
    init_timeHdlr.start_time = init_time_ptr;
    init_timeHdlr.name = (char *)malloc(NAME_MAX+1);
    strcpy(init_timeHdlr.name, name);

    return init_timeHdlr;
}


// Delete the shared memory and global init_time
// Call this function only when no tasks are using the shared resource
int global_init_time_destroy(global_init_time_t init_timeHdlr) {
    if (munmap((void *)init_timeHdlr.start_time, sizeof(struct timespec))) {
        perror("munmap");
        //return -1;
    }
    init_timeHdlr.start_time = NULL;
    if (close(init_timeHdlr.shm_fd)) {
        perror("close");
        //return -1;
    }
    init_timeHdlr.shm_fd = 0;
    if (shm_unlink(init_timeHdlr.name)) {
        perror("shm_unlink");
        //return -1;
    }
    free(init_timeHdlr.name);
    return 0;
}
