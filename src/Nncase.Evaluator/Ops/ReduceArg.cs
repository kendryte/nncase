using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        private torch.Tensor VisitReduceArg(ReduceArg reduceArg)
        {
            var input = _context.GetTorchArgument(reduceArg, ReduceArg.Input);
            var axis = _context.GetArgumentConst(reduceArg, ReduceArg.Axis).ToScalar<int>();
            var keepDims = _context.GetArgumentConst(reduceArg, ReduceArg.KeepDims).ToScalar<bool>();
            var selectLastIndex = _context.GetArgumentConst(reduceArg, ReduceArg.SelectLastIndex).ToScalar<bool>();
            if (selectLastIndex)
            {
                throw new NotImplementedException();
            }
            else
            {
                return reduceArg.ReduceArgOp switch
                {
                    ReduceArgOp.ArgMax => input.argmax(axis, keepDims),
                    ReduceArgOp.ArgMin => input.argmin(axis, keepDims),
                    _ => throw new NotSupportedException("Not Supported ReduceArgOp")
                };
            }
        }
    }
}