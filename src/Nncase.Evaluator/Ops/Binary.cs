using System;
using Nncase.IR.Math;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitBinary(Binary binary)
        {
            var a = _context.GetParam(0);
            var b = _context.GetParam(1);
            return binary.BinaryOp switch
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
            };
        }
    }
}