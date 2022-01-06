using System;
using Nncase.IR.Math;
using TorchSharp;
using static Tensorflow.Binding;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitBinary(Binary binary)
        {
            var res = tf.constant(1) * tf.constant(2);
            var a = _context.GetTorchArgument(binary, Binary.Lhs);
            var b = _context.GetTorchArgument(binary, Binary.Rhs);
            return (binary.BinaryOp switch
            {
                BinaryOp.Add => a + b,
                BinaryOp.Sub => a - b,
                BinaryOp.Mul => a * b,
                BinaryOp.Div => a / b,
                BinaryOp.Mod => a % b,
                BinaryOp.Min => torch.minimum(a, b),
                BinaryOp.Max => torch.maximum(a, b),
                BinaryOp.Pow => torch.pow(a, b),
                BinaryOp.BitwiseAnd => torch.bitwise_and(a, b),
                BinaryOp.BitwiseOr => torch.bitwise_or(a, b),
                BinaryOp.BitwiseXor => torch.bitwise_xor(a, b),
                BinaryOp.LogicalAnd => torch.logical_and(a, b),
                BinaryOp.LogicalOr => torch.logical_or(a, b),
                BinaryOp.LogicalXor => torch.logical_xor(a, b),
                _ => throw new ArgumentOutOfRangeException()
            }).to_type(_context.CurrentCall.CheckedDataType.ToTorchType());
        }
    }
}