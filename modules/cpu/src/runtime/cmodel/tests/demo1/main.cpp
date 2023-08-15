#include "cluster_def.h"
// #include <pthread.h>

#define DEFINE_TFUNC(b, t)                                                     \
    void *f_##b##_##t(void *arg) {                                             \
        block##b::thread##t::stage1_kernel(WK);                                \
        return arg;                                                            \
    }

#define DEFINE_BFUNC(b)                                                        \
    DEFINE_TFUNC(b, 0)                                                         \
    DEFINE_TFUNC(b, 1)                                                         \
    DEFINE_TFUNC(b, 2)                                                         \
    DEFINE_TFUNC(b, 3)

tensor<float, tensor_loc_t::device> WK({64, 8192, 128});

DEFINE_BFUNC(0)
DEFINE_BFUNC(1)
DEFINE_BFUNC(2)
DEFINE_BFUNC(3)
DEFINE_BFUNC(4)
DEFINE_BFUNC(5)
DEFINE_BFUNC(6)
DEFINE_BFUNC(7)

int main() {
    global_hardware_init();

    pthread_t t_0_0; // t_1_0, t_2_0, t_3_0, t_4_0, t_5_0, t_6_0, t_7_0;
    pthread_t t_0_1; // t_1_1, t_2_1, t_3_1, t_4_1, t_5_1, t_6_1, t_7_1;
    pthread_t t_0_2; // t_1_2, t_2_2, t_3_2, t_4_2, t_5_2, t_6_2, t_7_2;
    pthread_t t_0_3; // t_1_3, t_2_3, t_3_3, t_4_3, t_5_3, t_6_3, t_7_3;

    pthread_create(&t_0_0, NULL, f_0_0, NULL);
    pthread_create(&t_0_1, NULL, f_0_1, NULL);
    pthread_create(&t_0_2, NULL, f_0_2, NULL);
    pthread_create(&t_0_3, NULL, f_0_3, NULL);
    // pthread_create(&t_0_2, NULL, (void *(*)(void *)) & func_0_2, NULL);
    // pthread_create(&t_0_3, NULL, (void *(*)(void *)) & func_0_3, NULL);
    // pthread_create(&t_1_0, NULL, (void *(*)(void *)) & func_1_0, NULL);
    // pthread_create(&t_1_1, NULL, (void *(*)(void *)) & func_1_1, NULL);
    // pthread_create(&t_1_2, NULL, (void *(*)(void *)) & func_1_2, NULL);
    // pthread_create(&t_1_3, NULL, (void *(*)(void *)) & func_1_3, NULL);
    // pthread_create(&t_2_0, NULL, (void *(*)(void *)) & func_2_0, NULL);
    // pthread_create(&t_2_1, NULL, (void *(*)(void *)) & func_2_1, NULL);
    // pthread_create(&t_2_2, NULL, (void *(*)(void *)) & func_2_2, NULL);
    // pthread_create(&t_2_3, NULL, (void *(*)(void *)) & func_2_3, NULL);
    // pthread_create(&t_3_0, NULL, (void *(*)(void *)) & func_3_0, NULL);
    // pthread_create(&t_3_1, NULL, (void *(*)(void *)) & func_3_1, NULL);
    // pthread_create(&t_3_2, NULL, (void *(*)(void *)) & func_3_2, NULL);
    // pthread_create(&t_3_3, NULL, (void *(*)(void *)) & func_3_3, NULL);
    // pthread_create(&t_4_0, NULL, (void *(*)(void *)) & func_4_0, NULL);
    // pthread_create(&t_4_1, NULL, (void *(*)(void *)) & func_4_1, NULL);
    // pthread_create(&t_4_2, NULL, (void *(*)(void *)) & func_4_2, NULL);
    // pthread_create(&t_4_3, NULL, (void *(*)(void *)) & func_4_3, NULL);
    // pthread_create(&t_5_0, NULL, (void *(*)(void *)) & func_5_0, NULL);
    // pthread_create(&t_5_1, NULL, (void *(*)(void *)) & func_5_1, NULL);
    // pthread_create(&t_5_2, NULL, (void *(*)(void *)) & func_5_2, NULL);
    // pthread_create(&t_5_3, NULL, (void *(*)(void *)) & func_5_3, NULL);
    // pthread_create(&t_6_0, NULL, (void *(*)(void *)) & func_6_0, NULL);
    // pthread_create(&t_6_1, NULL, (void *(*)(void *)) & func_6_1, NULL);
    // pthread_create(&t_6_2, NULL, (void *(*)(void *)) & func_6_2, NULL);
    // pthread_create(&t_6_3, NULL, (void *(*)(void *)) & func_6_3, NULL);
    // pthread_create(&t_7_0, NULL, (void *(*)(void *)) & func_7_0, NULL);
    // pthread_create(&t_7_1, NULL, (void *(*)(void *)) & func_7_1, NULL);
    // pthread_create(&t_7_2, NULL, (void *(*)(void *)) & func_7_2, NULL);
    // pthread_create(&t_7_3, NULL, (void *(*)(void *)) & func_7_3, NULL);
    pthread_join(t_0_0, NULL);
    pthread_join(t_0_1, NULL);
    pthread_join(t_0_2, NULL);
    pthread_join(t_0_3, NULL);

    return 0;
}