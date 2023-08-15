#include "cluster_def.h"
// #include <pthread.h>

#define DEFINE_THREAD(b, t)                                                    \
    [[maybe_unused]] std::function<void *(void *)> func_##b##_##t =            \
        [&WK](void *) -> void * {                                              \
        block##b::thread##t::stage1_kernel(WK);                                \
        return nullptr;                                                        \
    };                                                                         \
    [[maybe_unused]] pthread_t t_##b##_##t

int main() {
    global_hardware_init();

    // tensor<float, tensor_loc_t::device> WQ({64, 8192, 128});
    tensor<float, tensor_loc_t::device> WK({64, 8192, 128});
    // tensor<float, tensor_loc_t::device> WV({64, 8192, 128});
    // tensor<float, tensor_loc_t::device> WFC1({384, 8192});

    DEFINE_THREAD(0, 0);
    DEFINE_THREAD(0, 1);
    DEFINE_THREAD(0, 2);
    DEFINE_THREAD(0, 3);
    DEFINE_THREAD(1, 0);
    DEFINE_THREAD(1, 1);
    DEFINE_THREAD(1, 2);
    DEFINE_THREAD(1, 3);
    DEFINE_THREAD(2, 0);
    DEFINE_THREAD(2, 1);
    DEFINE_THREAD(2, 2);
    DEFINE_THREAD(2, 3);
    DEFINE_THREAD(3, 0);
    DEFINE_THREAD(3, 1);
    DEFINE_THREAD(3, 2);
    DEFINE_THREAD(3, 3);
    DEFINE_THREAD(4, 0);
    DEFINE_THREAD(4, 1);
    DEFINE_THREAD(4, 2);
    DEFINE_THREAD(4, 3);
    DEFINE_THREAD(5, 0);
    DEFINE_THREAD(5, 1);
    DEFINE_THREAD(5, 2);
    DEFINE_THREAD(5, 3);
    DEFINE_THREAD(6, 0);
    DEFINE_THREAD(6, 1);
    DEFINE_THREAD(6, 2);
    DEFINE_THREAD(6, 3);
    DEFINE_THREAD(7, 0);
    DEFINE_THREAD(7, 1);
    DEFINE_THREAD(7, 2);
    DEFINE_THREAD(7, 3);

    pthread_create(&t_0_0, NULL, (void *(*)(void *)) & func_0_0, NULL);
    pthread_create(&t_0_1, NULL, (void *(*)(void *)) & func_0_1, NULL);
    pthread_create(&t_0_2, NULL, (void *(*)(void *)) & func_0_2, NULL);
    pthread_create(&t_0_3, NULL, (void *(*)(void *)) & func_0_3, NULL);
    pthread_create(&t_1_0, NULL, (void *(*)(void *)) & func_1_0, NULL);
    pthread_create(&t_1_1, NULL, (void *(*)(void *)) & func_1_1, NULL);
    pthread_create(&t_1_2, NULL, (void *(*)(void *)) & func_1_2, NULL);
    pthread_create(&t_1_3, NULL, (void *(*)(void *)) & func_1_3, NULL);
    pthread_create(&t_2_0, NULL, (void *(*)(void *)) & func_2_0, NULL);
    pthread_create(&t_2_1, NULL, (void *(*)(void *)) & func_2_1, NULL);
    pthread_create(&t_2_2, NULL, (void *(*)(void *)) & func_2_2, NULL);
    pthread_create(&t_2_3, NULL, (void *(*)(void *)) & func_2_3, NULL);
    pthread_create(&t_3_0, NULL, (void *(*)(void *)) & func_3_0, NULL);
    pthread_create(&t_3_1, NULL, (void *(*)(void *)) & func_3_1, NULL);
    pthread_create(&t_3_2, NULL, (void *(*)(void *)) & func_3_2, NULL);
    pthread_create(&t_3_3, NULL, (void *(*)(void *)) & func_3_3, NULL);
    pthread_create(&t_4_0, NULL, (void *(*)(void *)) & func_4_0, NULL);
    pthread_create(&t_4_1, NULL, (void *(*)(void *)) & func_4_1, NULL);
    pthread_create(&t_4_2, NULL, (void *(*)(void *)) & func_4_2, NULL);
    pthread_create(&t_4_3, NULL, (void *(*)(void *)) & func_4_3, NULL);
    pthread_create(&t_5_0, NULL, (void *(*)(void *)) & func_5_0, NULL);
    pthread_create(&t_5_1, NULL, (void *(*)(void *)) & func_5_1, NULL);
    pthread_create(&t_5_2, NULL, (void *(*)(void *)) & func_5_2, NULL);
    pthread_create(&t_5_3, NULL, (void *(*)(void *)) & func_5_3, NULL);
    pthread_create(&t_6_0, NULL, (void *(*)(void *)) & func_6_0, NULL);
    pthread_create(&t_6_1, NULL, (void *(*)(void *)) & func_6_1, NULL);
    pthread_create(&t_6_2, NULL, (void *(*)(void *)) & func_6_2, NULL);
    pthread_create(&t_6_3, NULL, (void *(*)(void *)) & func_6_3, NULL);
    pthread_create(&t_7_0, NULL, (void *(*)(void *)) & func_7_0, NULL);
    pthread_create(&t_7_1, NULL, (void *(*)(void *)) & func_7_1, NULL);
    pthread_create(&t_7_2, NULL, (void *(*)(void *)) & func_7_2, NULL);
    pthread_create(&t_7_3, NULL, (void *(*)(void *)) & func_7_3, NULL);

    return 0;
}