using Xunit;
using TorchSharp;
using System.Runtime.CompilerServices;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using System;
using static Nncase.Pattern.Utility;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.F.NN;
using static Nncase.Pattern.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Math;
using Rule = Nncase.Transform.Rule;
using Nncase.Transform;
using Nncase.TIR;
using Nncase.Pattern;
using Nncase.IR;
using Nncase.Evaluator;

namespace Nncase.Tests.ReWrite
{
    public sealed class SizeVarMul1Case : IRewriteCase
    {
        private SizeVar v = new("i");

        public override Expr PreExpr => (v * 1) * 2;
        public override Expr PostExpr => v * 2;

        public override IEnumerable<PatternRule> Rules => Nncase.Transform.DataFlow.Rules.SimplifyFactory.SimplifyMul().Take(1);
    }
}
