using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using TorchSharp;

using torchF = TorchSharp.torch.nn.functional;
namespace Nncase.Evaluator.Ops
{
    public sealed partial class EvaluatorVisitor
    {
        // private torch.Tensor VisitGatherND(GatherND gatherND)
        // {
        //     var input = _context.GetArgument(gatherND, GatherND.Input);
        //     var index = _context.GetArgumentConst(gatherND, GatherND.Index).ToArray<int>();
        //     var batchDims = _context.GetArgumentConst(gatherND, GatherND.BatchDims).ToScalar<int>();
        //     
        // }
    }
}