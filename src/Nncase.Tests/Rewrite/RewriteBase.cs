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
    public static class DummyOp
    {
        public static Call Conv2D(Expr input, int in_channels, int out_channels, int kernel = 3, int stride = 1)
        {
            var weights = torch.rand(new long[] { (long)out_channels, (long)in_channels, (long)kernel, (long)kernel }).ToConst();
            var bias = torch.rand(new long[(long)out_channels]).ToConst();
            return IR.F.NN.Conv2D(input, weights, bias, new[] { stride, stride }, Const.FromSpan<int>(new[] { 1, 1, 1, 1 }, new[] { 2, 2 }), new[] { 1, 1 }, PadMode.Constant, 1);
        }
    }

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

    public abstract class IRewriteCase
    {
        public virtual string Name { get => "Test" + this.GetType().Name; }

        public virtual Expr PreExpr { get; }

        public virtual IEnumerable<PatternRule> Rules { get; }
    }

    public sealed class TransposeConstBinaryCase : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var c = torch.rand(1, 2, 3, 4).ToConst();
                var input = torch.rand(1, 3, 1, 2).ToConst();
                var b = NHWCToNCHW(input) + c;
                return b;
            }
        }

        public override IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.TransposeConstBinaryMotionLeft(),
          new Transform.Rule.TransposeConstBinaryMotionRight(),
        };
    }

    public class FoldTransposeCase : IRewriteCase
    {
        public override Expr PreExpr
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

        public override IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.FoldTranspose(),
        };
    }

    /// <summary>
    /// a simple noptranspose case, for the match test
    /// </summary>
    public class FoldNopTransposeCase1 : IRewriteCase
    {
        public override Expr PreExpr
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
        public override IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.FoldTranspose(),
        };
    }

    public class FoldNopTransposeCase2 : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 3, 1, 2).ToConst();
                var b = NHWCToNCHW(input);
                var rhs = b + torch.rand(1, 2, 3, 4).ToConst();
                var lhs = NCHWToNHWC(b) - torch.rand(1, 3, 1, 2).ToConst();
                var e = lhs + NCHWToNHWC(rhs);
                return e;
            }
        }
        public override IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.FoldTranspose(),
          new Transform.Rule.FoldNopTranspose(),
        };
    }

    public sealed class FoldNopTransposeCase3 : FoldNopTransposeCase2
    {
        public override IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.FoldTranspose(),
          new Transform.Rule.FoldNopTranspose(),
          new Transform.Rule.TransposeBinaryMotion(),
          new Transform.Rule.TransposeConstBinaryMotionLeft(),
          new Transform.Rule.TransposeConstBinaryMotionRight(),
        };
    }

    /// <summary>
    /// transpose demo
    /// </summary>
    public sealed class TransposeDemoCase : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 28, 28, 3).ToConst();
                var conv1 = NCHWToNHWC(DummyOp.Conv2D(NCHWToNHWC(input), 3, out_channels: 8, 3, 2));
                var lhs = NCHWToNHWC(DummyOp.Conv2D(NCHWToNHWC(conv1), 8, out_channels: 3, 3, 1));
                var rhs = conv1 + torch.rand(new long[] { 1, 28, 1, 3 }).ToConst();
                return lhs + rhs;
            }
        }
        public override IEnumerable<PatternRule> Rules => new PatternRule[]{
          new Transform.Rule.FoldTranspose(),
          new Transform.Rule.FoldNopTranspose(),
          new Transform.Rule.TransposeBinaryMotion(),
          new Transform.Rule.TransposeConstBinaryMotionLeft(),
          new Transform.Rule.TransposeConstBinaryMotionRight(),
        };
    }

}