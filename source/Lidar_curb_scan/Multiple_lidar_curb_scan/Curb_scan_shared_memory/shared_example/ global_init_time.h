#ifndef GLOBAL_INIT_TIME_H
#define GLOBAL_INIT_TIME_H

#include <sys/time.h>

#define GLOBAL_INIT_TIME_SHARED_MEMORY_NAME "Global_Init_Time_name"

// Structure of a global init time.
typedef struct global_init_time_t {
    struct timespec *start_time;
    int shm_fd;
    char* name;
    int created;
} global_init_time_t;


global_init_time_t global_init_time_init(char *name);
int global_init_time_destroy(global_init_time_t init_timeHdlr);

#endif // GLOBAL_INIT_TIME_H