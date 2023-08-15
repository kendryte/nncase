#include <hardware_context.h>
#include <nncase/compiler_defs.h>
#include <tdma.h>

namespace block0 {
#include "shared_def.h"
constexpr size_t bid = 0;
namespace thread0 {
constexpr size_t tid = 0;
#include "kernel.h"
} // namespace thread0
namespace thread1 {
constexpr size_t tid = 1;
#include "kernel.h"
} // namespace thread1
namespace thread2 {
constexpr size_t tid = 2;
#include "kernel.h"
} // namespace thread2
namespace thread3 {
constexpr size_t tid = 3;
#include "kernel.h"
} // namespace thread3
} // namespace block0

// namespace block1 {
// namespace thread0 {}
// namespace thread1 {}
// namespace thread2 {}
// namespace thread3 {}
// } // namespace block1
// namespace block2 {
// namespace thread0 {}
// namespace thread1 {}
// namespace thread2 {}
// namespace thread3 {}
// } // namespace block2
// namespace block3 {
// namespace thread0 {}
// namespace thread1 {}
// namespace thread2 {}
// namespace thread3 {}
// } // namespace block3
// namespace block4 {
// namespace thread0 {}
// namespace thread1 {}
// namespace thread2 {}
// namespace thread3 {}
// } // namespace block4
// namespace block5 {
// namespace thread0 {}
// namespace thread1 {}
// namespace thread2 {}
// namespace thread3 {}
// } // namespace block5
// namespace block6 {
// namespace thread0 {}
// namespace thread1 {}
// namespace thread2 {}
// namespace thread3 {}
// } // namespace block6
// namespace block7 {
// namespace thread0 {}
// namespace thread1 {}
// namespace thread2 {}
// namespace thread3 {}
// } // namespace block7

int main() {
    tensor<float, tensor_loc_t::device> WQ({64, 8192, 128});
    tensor<float, tensor_loc_t::device> WK({64, 8192, 128});
    tensor<float, tensor_loc_t::device> WV({64, 8192, 128});
    tensor<float, tensor_loc_t::device> WFC1({384, 8192});
    block0::thread0::stage1_kernel(WQ, WK, WV, WFC1);
    return 0;
}