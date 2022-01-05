using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Tensors;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private static Tensor one_hot(
            Tensor indices,
            Tensor depth,
            Tensor on_value = null,
            Tensor off_value = null,
            TF_DataType dtype = TF_DataType.DtInvalid,
            int axis = -1,
            string name = null)
        {
            return Binding.tf_with<ops.NameScope, Tensor>(ops.name_scope(name, nameof(one_hot), (object) new
            {
                indices = indices,
                depth = depth,
                dtype = dtype
            }), (Func<ops.NameScope, Tensor>) (scope =>
            {
                TF_DataType tfDataType1 = TF_DataType.DtInvalid;
                TF_DataType tfDataType2 = TF_DataType.DtInvalid;
                if (dtype == TF_DataType.DtInvalid)
                    dtype = TF_DataType.TF_FLOAT;
                on_value = ops.convert_to_tensor((object) on_value, dtype, nameof(on_value));
                tfDataType1 = dtype;
                off_value = ops.convert_to_tensor((object) off_value, dtype, name = nameof(off_value));
                tfDataType2 = dtype;
                return gen_array_ops.one_hot(indices, depth, on_value, off_value, axis: axis, name: name);
            }));
        }

        private Tensorflow.Tensor VisitOneHot(OneHot oneHot)
        {
            var depth = _context.GetArgumentConstScalar<int>(oneHot, OneHot.Depth);
            var rawIndices = _context.GetTFArgument(oneHot, OneHot.Indices);
            var afterIndices = rawIndices.ToConst().ToArray<int>().Select(x => x < 0 ? x + depth : x).ToArray();
            var indices = new NDArray(afterIndices, rawIndices.shape);
            var onValue = _context.GetTFArgument(oneHot, OneHot.OnValue);
            var offValue = _context.GetTFArgument(oneHot, OneHot.OffValue);
            var axis = _context.GetArgumentConstScalar<int>(oneHot, OneHot.Axis);
            return one_hot(
                indices, 
                ops.convert_to_tensor((object) depth),
                onValue,
                offValue,
                TF_DataType.TF_FLOAT, 
                axis);
        }
    }
}