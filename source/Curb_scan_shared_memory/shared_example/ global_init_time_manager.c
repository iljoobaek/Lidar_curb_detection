#include "global_init_time.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sched.h>
#include <pthread.h>
#include <string.h>


 int main(int argc, char *argv[]) 
{
    global_init_time_t init_time;
    int option = 0;

    /* To handle user input */
    if (argc > 1) {
        option = atoi(argv[1]);
    }
    else {
        printf("<usage>: ./xxx <option>\n");
        return 0;
    }

    init_time = global_init_time_init(GLOBAL_INIT_TIME_SHARED_MEMORY_NAME);
    if (init_time.start_time == NULL) {
        return 1;
    }

    if (init_time.created) {
        printf("[Global__Init_time] The init_time was just created\n");
    }

    printf("[Global__Init_time] got initial time : %llu\n", init_time.start_time->tv_sec * 1000000000ULL + init_time.start_time->tv_nsec);

    switch(option) {
        case 0:
            printf("[Global__Init_time] Destroy\n");
            if (global_init_time_destroy(init_time)) {
                printf("[Global__Init_time] Destroy Failed\n");
                return 1;
            }
            break;
        default:
            printf("[Global__Init_time] Wrong option %d\n", option);
            break;
    }

    return 0;
 }

int cleanup(global_init_time_t init_time)
{
  // Mutex destruction completely cleans it from system memory.
  if (global_init_time_destroy(init_time)) {
      return 1;
  }
  return 0;
}