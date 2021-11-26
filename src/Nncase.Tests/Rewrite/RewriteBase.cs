using Xunit;
using System;
using System.Collections.Generic;
using Nncase.IR;
using Nncase.Transform;
using Nncase.Pattern;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.NN;
using Rule = Nncase.Transform.Rule;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.NN;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.Utility;
using System.IO;
using System.Runtime.CompilerServices;
using TorchSharp;
using Nncase.Evaluator;


namespace Nncase.Tests.ReWrite
{
    public class RewriteTest
    {
        public RunPassOptions passOptions;

        private static string GetThisFilePath([CallerFilePath] string path = null)
        {
            return path;
        }

        public RewriteTest()
        {
            var TestName = this.GetType().Name;
            string dumpDir = Path.Combine(GetThisFilePath(), "..", "..", "..", "..", "tests_output");
            dumpDir = Path.GetFullPath(dumpDir);
            Directory.CreateDirectory(dumpDir);
            passOptions = new RunPassOptions(null, 3, dumpDir);
        }
        
        public Expr RunShapeInferPass(Expr expr, params Expr[] parameters)
        {
            var f = new Function(expr, parameters);
            TypeInference.InferenceType(f);
            return new ShapeInferPass().Run(f, passOptions.SetName("RunShapeInferPass")).Body;
        }
        
        public Expr ApplyFoldConstCallRewrite(Expr expr) =>
            DataFlowRewrite.Rewrite(expr, new Transform.DataFlow.Rules.FoldConstCall());
    }

    public interface IRewriteCase
    {
        public string Name { get => "Test" + this.GetType().Name; }

        public Expr PreExpr { get; }

        public IEnumerable<PatternRule> Rules { get; }
    }

    public sealed class TransposeConstBinaryCase : IRewriteCase
    {
        public Expr PreExpr
        {
            get
            {
                var c = torch.rand(1, 2, 3, 4).ToConst();
                var input = torch.rand(1, 3, 1, 2).ToConst();
                var b = NHWCToNCHW(input) + c;
                return b;
            }
        }

        public IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.TransposeConstBinaryMotionLeft(),
          new Transform.Rule.TransposeConstBinaryMotionRight(),
        };
    }

    public class FoldTransposeCase : IRewriteCase
    {
        public Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 3, 1, 2).ToConst();
                var b = NHWCToNCHW(input); // [1,2,3,1]
                var d = NCHWToNHWC(b) * torch.rand(1, 3, 4, 2).ToConst();
                var e = d + 100.0f;
                return e;
            }
        }

        public IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.FoldTranspose(),
        };
    }

    /// <summary>
    /// a simple noptranspose case, for the match test
    /// </summary>
    public sealed class FoldNopTransposeCase1 : IRewriteCase
    {
        public Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 3, 1, 2).ToConst();
                var b = Transpose(input, new[] { 0, 1, 2, 3 }); // [1,2,3,1]
                var d = NCHWToNHWC(b) * torch.rand(1, 1, 2, 1).ToConst();
                var e = d + 100.0f;
                return e;
            }
        }
        public IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.FoldTranspose(),
        };
    }

    public sealed class FoldNopTransposeCase2 : FoldTransposeCase
    {
        public IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.FoldTranspose(),
          new Transform.Rule.FoldNopTranspose(),
        };
    }

}