#include "thread_pool.h"
#include <stdexcept>

using namespace nncase::runtime::cpu::thread_pool;

int threads_size =
    atoi(getenv("NNCASE_MAX_THREADS") ? getenv("NNCASE_MAX_THREADS") : "0");
int threads_count;
std::vector<pthread_t> threads;
std::vector<void *> users;
uintptr_t nncase::runtime::cpu::thread_pool::paddr_offset;

void *nncase::runtime::cpu::thread_pool::thread_start(thread_func callable,
                                                      void *user,
                                                      size_t user_size) {
    auto user_ = malloc(user_size);
    std::memcpy(user_, user, user_size);
    thread_func new_call = thread_func((char *)callable + paddr_offset);
    if (threads_size == 0) {
        new_call(user_);
    } else {
        auto idx = threads_count % threads_size;
        if (threads_count >= threads_size) {
            pthread_join(threads[idx], NULL);
            free(users[idx]);
        }
        pthread_t pt;
        auto ret = pthread_create(&pt, NULL, new_call, user_);
        if (ret != 0) {
            throw std::runtime_error("thread creation failed\n");
        }

        if (threads_count == 0) {
            threads.resize(threads_size);
            users.resize(threads_size);
        }
        threads[idx] = pt;
        users[idx] = user_;
        threads_count++;
    }
    return nullptr;
}

void *nncase::runtime::cpu::thread_pool::thread_end() {
    if (threads_size) {
        for (int i = 0; i < std::min(threads_size, threads_count); i++) {
            // if (threads[i].joinable()) {
            pthread_join(threads[i], NULL);
            free(users[i]);
            // }
        }
        threads_count = 0;
        threads.clear();
        users.clear();
    }
    return nullptr;
}
