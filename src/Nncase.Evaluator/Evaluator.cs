using System;
using System.Linq;
using Nncase.Evaluator.Ops;
using Nncase.IR;
using Nncase.IR.Math;
using TorchSharp;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using static TorchSharp.TensorExtensionMethods;
using Sigmoid = Nncase.IR.NN.Sigmoid;

namespace Nncase.Evaluator
{
    public static class Evaluator
    {
        public static torch.Tensor Eval(Expr expr)
        {
            var evaluatorVisitor = new EvaluatorVisitor();
            return evaluatorVisitor.Visit(expr);
        }
    }
}