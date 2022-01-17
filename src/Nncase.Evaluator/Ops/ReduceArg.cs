using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;
using Nncase.IR;

namespace Nncase.Evaluator.Ops
{
    public class ReduceArgEvaluator : IEvaluator<ReduceArg>
    {
        public static Const Visit(EvaluatorContext context, ReduceArg reduceArg)
        {
            var input = context.GetTorchArgument(reduceArg, ReduceArg.Input);
            var axis = context.GetArgumentConst(reduceArg, ReduceArg.Axis).ToScalar<int>();
            var keepDims = context.GetArgumentConst(reduceArg, ReduceArg.KeepDims).ToScalar<bool>();
            var selectLastIndex = context.GetArgumentConst(reduceArg, ReduceArg.SelectLastIndex).ToScalar<bool>();
            if (selectLastIndex)
            {
                throw new NotImplementedException();
            }
            else
            {
                return (reduceArg.ReduceArgOp switch
                {
                    ReduceArgOp.ArgMax => input.argmax(axis, keepDims),
                    ReduceArgOp.ArgMin => input.argmin(axis, keepDims),
                    _ => throw new NotSupportedException("Not Supported ReduceArgOp")
                }).ToConst();
            }
        }
    }
}