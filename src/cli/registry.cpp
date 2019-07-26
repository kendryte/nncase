namespace nncase
{
namespace codegen
{
    void register_netural_emitters();
}
}

namespace nncase
{
namespace ir
{
    void register_neutral_evaluators();
}
}

void init_codegen_ops()
{
    using namespace nncase::codegen;

    register_netural_emitters();
}

void init_evaluator_ops()
{
    using namespace nncase::ir;

    register_neutral_evaluators();
}
