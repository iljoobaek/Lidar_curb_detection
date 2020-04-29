
INCLUDES= $(HOME)/work/Lock-based-sched/global_init_time

EXTRA_CFLAGS= -I$(INCLUDES)

EXTRA_LDFLAGS= -lrt -lpthread 


CC=gcc
all: global_init_time_manager.c
	$(CC) global_init_time_manager.c global_init_time.c -o global_init_time_manager $(EXTRA_CFLAGS) $(EXTRA_LDFLAGS)

clean:
	rm global_init_time_manager