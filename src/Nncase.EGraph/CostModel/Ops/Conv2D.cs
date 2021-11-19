using Nncase;
using Nncase.IR.Math;
using Nncase.IR.NN;

namespace Nncase.CostModel
{
    public sealed partial class ExprCostModelVisitor
    {
        private Cost VisitConv2D(Conv2D conv2D)
        {
            // https://stackoverflow.com/questions/56138754/formula-to-compute-the-number-of-macs-in-a-convolutional-neural-network
            var input = _context.GetArgumentType(conv2D, Conv2D.Input).Shape;
            var weights = _context.GetArgumentType(conv2D, Conv2D.Weights).Shape;
            var output = _context.CurrentCallResultTensorType().Shape;
            // weights: [output, input, H, W]
            var arithm = (weights.Prod() * input[0] * output[2] * output[3]).FixedValue;
            return new Cost(arithm);
        }
    }
}