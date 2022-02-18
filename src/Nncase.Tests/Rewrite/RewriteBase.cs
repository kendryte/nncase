using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Hosting;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Pattern;
using Nncase.Transform;
using TorchSharp;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.Pattern.F.Math;
using static Nncase.Pattern.F.NN;
using static Nncase.Pattern.F.Tensors;
using static Nncase.Pattern.Utility;
using Rule = Nncase.Transform.Rule;


namespace Nncase.Tests.ReWriteTest
{
    public static class DummyOp
    {
        public static Call Conv2D(Expr input, int in_channels, int out_channels, int kernel = 3, int stride = 1)
        {
            var weights = torch.rand(new long[] { (long)out_channels, (long)in_channels, (long)kernel, (long)kernel }).ToTensor();
            var bias = torch.rand(new long[] { (long)out_channels }).ToTensor();
            return IR.F.NN.Conv2D(input, weights, bias, new[] { stride, stride }, Tensor.FromSpan<int>(new[] { 1, 1, 1, 1 }, new[] { 2, 2 }), new[] { 1, 1 }, PadMode.Constant, 1);
        }
    }

    public class RewriteFixtrue : IHostFixtrue
    {
        public RunPassOptions passOptions;

        private static string GetThisFilePath([CallerFilePath] string path = null)
        {
            return path;
        }

        public RewriteFixtrue(IHost host) : base(host)
        {
            var TestName = this.GetType().Name;
            string dumpDir = Path.Combine(GetThisFilePath(), "..", "..", "..", "..", "tests_output");
            dumpDir = Path.GetFullPath(dumpDir);
            Directory.CreateDirectory(dumpDir);
            passOptions = new RunPassOptions(null, 3, dumpDir);
        }

        public Expr RunShapeInferPass(string name, Expr expr, params Expr[] parameters)
        {
            expr.InferenceType();
            var f = new Function(expr, parameters);
            var result = CompilerServices.InferenceType(f);
            f.DumpExprAsIL("before", Path.Combine(passOptions.PassDumpDir, $"ShapeInfer_{name}"));
            return new ShapeInferPass($"ShapeInfer_{name}").Run(f, passOptions).Body;
        }

        public Expr ApplyFoldConstCallRewrite(Expr expr) =>
            DataFlowRewrite.Rewrite(expr, new[] { new Transform.Rule.FoldConstCall() }, passOptions);
    }

    public abstract class IRewriteCase
    {
        /// <summary>
        /// Get Name
        /// </summary>
        public virtual string Name { get => "Test" + this.GetType().Name; }

        /// <summary>
        /// Get Pre Expr
        /// </summary>
        public virtual Expr PreExpr { get; }

        /// <summary>
        /// Get Post Expr
        /// </summary>
        public virtual Expr PostExpr { get; }

        /// <summary>
        /// get rules
        /// </summary>
        public virtual IEnumerable<IRewriteRule> Rules { get; }

        /// <summary>
        /// the eval inputs dict
        /// </summary>
        public virtual Dictionary<Var, torch.Tensor> Inputs { get; } = new();
    }

    public sealed class TransposeConstBinaryCase : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var c = torch.rand(1, 2, 3, 4).ToTensor();
                var input = torch.rand(1, 3, 1, 2).ToTensor();
                var b = NHWCToNCHW(input) + c;
                return b;
            }
        }

        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
          new Transform.Rule.TransposeConstBinaryMotionLeft(),
          new Transform.Rule.TransposeConstBinaryMotionRight(),
        };
    }

    public sealed class FoldReshapeCase : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 3, 1, 2).ToTensor();
                var b = Reshape(input, (Const)new[] { 1, 1, 1, 6 });
                var c = Reshape(input, (Const)new[] { 1, 1, 3, 2 });
                return c;
            }
        }

        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
          new Transform.Rule.FoldReshape(),
        };
    }
    public sealed class FoldNopReshapeCase : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 3, 1, 2).ToTensor();
                var b = Reshape(input, (Const)new[] { 1, 3, 1, 2 });
                return b;
            }
        }

        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
          new Transform.Rule.FoldNopReshape(),
        };
    }

    public sealed class FoldNopClampCase : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 3, 1, 2).ToTensor();
                var b = Clamp(input, float.MinValue, float.MaxValue);
                return b;
            }
        }

        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
          new Transform.Rule.FoldNopClamp(),
        };
    }

    public class FoldTransposeCase : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 3, 1, 2).ToTensor();
                var b = NHWCToNCHW(input); // [1,2,3,1]
                var d = NCHWToNHWC(b) * torch.rand(1, 3, 4, 2).ToTensor();
                var e = d + 100.0f;
                return e;
            }
        }

        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
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
                var input = torch.rand(1, 3, 1, 2).ToTensor();
                var b = Transpose(input, new[] { 0, 1, 2, 3 }); // [1,2,3,1]
                var d = NCHWToNHWC(b) * torch.rand(1, 1, 2, 1).ToTensor();
                var e = d + 100.0f;
                return e;
            }
        }
        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
          new Transform.Rule.FoldTranspose(),
        };
    }

    public class FoldNopTransposeCase2 : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 3, 1, 2).ToTensor();
                var b = NHWCToNCHW(input);
                var rhs = b + torch.rand(1, 2, 3, 4).ToTensor();
                var lhs = NCHWToNHWC(b) - torch.rand(1, 3, 1, 2).ToTensor();
                var e = lhs + NCHWToNHWC(rhs);
                return e;
            }
        }
        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
          new Transform.Rule.FoldTranspose(),
          new Transform.Rule.FoldNopTranspose(),
        };
    }

    public class FoldNopTransposeCase3 : FoldNopTransposeCase2
    {
        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
          new Transform.Rule.FoldTranspose(),
          new Transform.Rule.FoldNopTranspose(),
          new Transform.Rule.TransposeBinaryMotion(),
          new Transform.Rule.TransposeConstBinaryMotionLeft(),
          new Transform.Rule.TransposeConstBinaryMotionRight(),
        };
    }

    public class ClassicDemo : IRewriteCase
    {
        public override Expr PreExpr
        {
            get
            {
                var x = (Const)1234;
                return (x * 2) / 2;
            }
        }
        public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
          new Transform.Rule.Xmul2(),
          new Transform.Rule.Xmul1(),
          new Transform.Rule.ReassociateDiv(),
          new Transform.Rule.ReassociateMul(),
          new Transform.Rule.ReassociateXY(),
          new Transform.Rule.XDivX(),
        };
    }


    /// <summary>
    /// transpose demo
    /// </summary>
    public sealed class TransposeDemoCase : FoldNopTransposeCase3
    {
        public override Expr PreExpr
        {
            get
            {
                var input = torch.rand(1, 28, 28, 3).ToTensor();
                var conv1 = NCHWToNHWC(DummyOp.Conv2D(NHWCToNCHW(input), 3, out_channels: 8, 3, 2));
                var lhs = NCHWToNHWC(DummyOp.Conv2D(NHWCToNCHW(conv1), 8, out_channels: 8, 3, 1));
                var rhs = conv1 + torch.rand(new long[] { 1, 14, 14, 8 }).ToTensor();
                return lhs + rhs;
            }
        }
    }

}