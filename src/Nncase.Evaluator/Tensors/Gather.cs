using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using static Tensorflow.Binding;
using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Gather"/>.
/// </summary>
public class GatherEvaluator : IEvaluator<Gather>, ITypeInferencer<Gather>
{
    /// <inheritdoc/>
    public Const Visit(IEvaluateContext context, Gather gather)
    {
        var input = context.GetTFArgumentValue(gather, Gather.Input);
        var axis = context.GetArgumentValue(gather, Gather.Axis).ToScalar<int>();
        var index = context.GetTFArgumentValue(gather, Gather.Index);
        return tf.gather(input, index, axis: axis).ToConst();
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Gather target)
    {
        var input = context.CheckArgumentType<TensorType>(target, Gather.Input);
        var axis = context.CheckArgumentType<TensorType>(target, Gather.Axis);
        var index = context.CheckArgumentType<TensorType>(target, Gather.Index);
        return Visit(context, target, input, axis, index);
    }

    private IRType Visit(ITypeInferenceContext context, Gather target, TensorType input, TensorType axis, TensorType index)
    {
        if (context.GetArgument(target, Flatten.Axis) is Const axisValue)
        {
            var axisV = axisValue.ToScalar<int>();
            axisV = axisV < 0 ? axisV + input.Shape.Rank : axisV;

            // input_shape[:axis] + index_shape + input_shape[axis + 1:]
            var newShape = input.Shape.InsertAndClone(axisV, index.Shape);
            return new TensorType(input.DType, newShape);
        }
        else
        {
            return new InvalidType("Gather axis must be constant");
        }
    }
}
