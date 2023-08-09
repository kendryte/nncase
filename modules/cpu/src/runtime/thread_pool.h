#ifndef THREAD_POOL_
#define THREAD_POOL_
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <nncase/runtime/cpu/compiler_defs.h>
#include <pthread.h>
#include <utility>
#include <vector>

BEGIN_NS_NNCASE_RT_MODULE(cpu)
namespace thread_pool {

using thread_func = void *(*)(void *);

// static int threads_size = atoi(getenv("NNCASE_MAX_THREADS") ?
// getenv("NNCASE_MAX_THREADS") : "0"); static int threads_count; static
// std::vector<pthread_t> threads; static std::vector<void *> users;
extern uintptr_t paddr_offset;

void *thread_start(thread_func callable, void *user, size_t user_size);
void *thread_end();

} // namespace thread_pool
END_NS_NNCASE_RT_MODULE

#endif
