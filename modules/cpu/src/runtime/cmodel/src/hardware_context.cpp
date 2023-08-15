#include <hardware_context.h>
#include <hardware_def.h>
#include <pthread.h>

class condition_variable {
  public:
    condition_variable() {
        pthread_cond_init(&cond, NULL);
        pthread_mutex_init(&mutex, NULL);
    }

    pthread_mutex_t mutex;
    pthread_cond_t cond;
};

struct hardware_context_impl {
  public:
    hardware_context_impl() {
        global_cond = condition_variable();
        pthread_mutex_init(&global_mutex, NULL);

        for (size_t i = 0; i < BLOCKS; i++) {
            pthread_mutex_init(&block_mutexs[i], NULL);
            block_conds[i] = condition_variable();
            for (size_t j = 0; j < CORES; j++) {
                block_visited[i][j] = false;
                global_visited[i][j] = false;
            }
        }
    }

    void reset_block_visited(int bid) {
        for (size_t i = 0; i < CORES; i++) {
            block_visited[bid][i] = false;
        }
    }

    void reset_global_visited() {
        for (size_t i = 0; i < BLOCKS; i++) {
            for (size_t j = 0; j < CORES; j++) {
                global_visited[i][j] = false;
            }
        }
    }

    pthread_mutex_t global_mutex;
    condition_variable global_cond;
    bool global_visited[BLOCKS][CORES];
    pthread_mutex_t block_mutexs[BLOCKS];
    condition_variable block_conds[BLOCKS];
    bool block_visited[BLOCKS][CORES];
};

hardware_context::hardware_context() {
    impl_ = std::make_unique<hardware_context_impl>();
}

void hardware_context::lock_block(int bid) {
    pthread_mutex_lock(&impl_->block_mutexs[bid]);
}

void hardware_context::unlock_block(int bid) {
    pthread_mutex_unlock(&impl_->block_mutexs[bid]);
}

int hardware_context::mark_block_visit(int bid, int tid) {
    impl_->block_visited[bid][tid] = true;
    int visited = 0;
    for (size_t i = 0; i < CORES; i++) {
        visited += impl_->block_visited[bid][i] == true ? 1 : 0;
    }
    return visited;
}

void hardware_context::wait_block_sync(int bid, int visited) {
    pthread_mutex_lock(&impl_->block_conds[bid].mutex);
    if (visited == CORES) {
        impl_->reset_block_visited(bid);
        pthread_cond_broadcast(&impl_->block_conds[bid].cond);
    } else {
        pthread_cond_wait(&impl_->block_conds[bid].cond,
                          &impl_->block_conds[bid].mutex);
    }
    pthread_mutex_unlock(&impl_->block_conds[bid].mutex);
}

void hardware_context::lock_all() { pthread_mutex_lock(&impl_->global_mutex); }

void hardware_context::unlock_all() {
    pthread_mutex_unlock(&impl_->global_mutex);
}

int hardware_context::mark_all_visit(int bid, int tid) {
    impl_->global_visited[bid][tid] = true;
    int visited = 0;
    for (size_t i = 0; i < BLOCKS; i++) {
        for (size_t j = 0; j < CORES; j++) {
            visited += impl_->global_visited[i][j] == true ? 1 : 0;
        }
    }
    return visited;
}

void hardware_context::wait_all_sync(int visited) {
    pthread_mutex_lock(&impl_->global_cond.mutex);
    if (visited == BLOCKS * CORES) {
        impl_->reset_global_visited();
        pthread_cond_broadcast(&impl_->global_cond.cond);
    } else {
        pthread_cond_wait(&impl_->global_cond.cond, &impl_->global_cond.mutex);
    }
    pthread_mutex_unlock(&impl_->global_cond.mutex);
}

std::unique_ptr<hardware_context> global_hardware_ctx;

void global_hardware_init() {
    global_hardware_ctx = std::make_unique<hardware_context>();
}