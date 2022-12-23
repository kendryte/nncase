#include <nncase/functional/ops.h>

namespace nncase::F {
void dummy_useage(runtime::runtime_tensor &a) {
    square(a, dt_float32).unwrap_or_throw();
}
} // namespace nncase::F