using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Transform;
using Nncase.Transform.Passes;
using Nncase.Transform.Rules.Neutral;
using OrtKISharp;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using static Nncase.IR.F.Tensors;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.NN;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Tests.RewriteTest;

public static class DummyOp
{
    public static Call Conv2D(Expr input, int in_channels, int out_channels, int kernel = 3, int stride = 1)
    {
        var weights = OrtKI.Random(out_channels, in_channels, kernel, kernel).ToTensor();
        var bias = OrtKI.Random(out_channels).ToTensor();
        return IR.F.NN.Conv2D(input, weights, bias, new[] { stride, stride }, Tensor.From<int>(new[] { 1, 1, 1, 1 }, new[] { 2, 2 }), new[] { 1, 1 }, PadMode.Constant, 1);
    }
}

public class RewriteFixtrue : TestFixture.UnitTestFixtrue
{
    public async Task<Expr> RunShapeInferPass(string name, RunPassOptions caseOptions, Expr expr, params Var[] parameters)
    {
        expr.InferenceType();
        var f = new Function(expr, parameters);
        var result = CompilerServices.InferenceType(f);
        CompilerServices.DumpIR(f, "before", Path.Combine(caseOptions.PassDumpDir, $"ShapeInfer_{name}"));
        return ((Function)await new ShapeInferPass($"ShapeInfer_{name}").RunAsync(f, caseOptions)).Body;
    }

    public Expr ApplyFoldConstCallRewrite(Expr expr, RunPassOptions caseOptions) =>
        CompilerServices.Rewrite(expr, new[] { new FoldConstCall() }, caseOptions);
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
    public virtual Dictionary<Var, OrtKISharp.Tensor> Inputs { get; } = new();
}

// public sealed class TransposeConstBinaryCase : IRewriteCase
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var c = torch.rand(1, 2, 3, 4).ToTensor();
//             var input = torch.rand(1, 3, 1, 2).ToTensor();
//             var b = NHWCToNCHW(input) + c;
//             return b;
//         }
//     }

//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.TransposeConstBinaryMotionLeft(),
//       new Transform.Rules.TransposeConstBinaryMotionRight(),
//     };
// }

// public sealed class FoldReshapeCase : IRewriteCase
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var input = torch.rand(1, 3, 1, 2).ToTensor();
//             var b = Reshape(input, (Const)new[] { 1, 1, 1, 6 });
//             var c = Reshape(input, (Const)new[] { 1, 1, 3, 2 });
//             return c;
//         }
//     }

//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.FoldReshape(),
//     };
// }
// public sealed class FoldNopReshapeCase : IRewriteCase
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var input = torch.rand(1, 3, 1, 2).ToTensor();
//             var b = Reshape(input, (Const)new[] { 1, 3, 1, 2 });
//             return b;
//         }
//     }

//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.FoldNopReshape(),
//     };
// }

// public sealed class FoldNopClampCase : IRewriteCase
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var input = torch.rand(1, 3, 1, 2).ToTensor();
//             var b = Clamp(input, float.MinValue, float.MaxValue);
//             return b;
//         }
//     }

//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.FoldNopClamp(),
//     };
// }

// public class FoldTransposeCase : IRewriteCase
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var input = torch.rand(1, 3, 1, 2).ToTensor();
//             var b = NHWCToNCHW(input); // [1,2,3,1]
//             var d = NCHWToNHWC(b) * torch.rand(1, 3, 4, 2).ToTensor();
//             var e = d + 100.0f;
//             return e;
//         }
//     }

//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.FoldTranspose(),
//     };
// }


// public class FoldNopTransposeCase1 : IRewriteCase
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var input = torch.rand(1, 3, 1, 2).ToTensor();
//             var b = Transpose(input, new[] { 0, 1, 2, 3 }); // [1,2,3,1]
//             var d = NCHWToNHWC(b) * torch.rand(1, 1, 2, 1).ToTensor();
//             var e = d + 100.0f;
//             return e;
//         }
//     }
//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.FoldTranspose(),
//     };
// }

// public class FoldNopTransposeCase2 : IRewriteCase
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var input = torch.rand(1, 3, 1, 2).ToTensor();
//             var b = NHWCToNCHW(input);
//             var rhs = b + torch.rand(1, 2, 3, 4).ToTensor();
//             var lhs = NCHWToNHWC(b) - torch.rand(1, 3, 1, 2).ToTensor();
//             var e = lhs + NCHWToNHWC(rhs);
//             return e;
//         }
//     }
//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.FoldTranspose(),
//       new Transform.Rules.FoldNopTranspose(),
//     };
// }

// public class FoldNopTransposeCase3 : FoldNopTransposeCase2
// {
//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.FoldTranspose(),
//       new Transform.Rules.FoldNopTranspose(),
//       new Transform.Rules.TransposeBinaryMotion(),
//       new Transform.Rules.TransposeConstBinaryMotionLeft(),
//       new Transform.Rules.TransposeConstBinaryMotionRight(),
//     };
// }

// public class ClassicDemo : IRewriteCase
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var x = (Const)1234;
//             return (x * 2) / 2;
//         }
//     }
//     public override IEnumerable<IRewriteRule> Rules => new IRewriteRule[]{
//       new Transform.Rules.Xmul2(),
//       new Transform.Rules.Xmul1(),
//       new Transform.Rules.ReassociateDiv(),
//       new Transform.Rules.ReassociateMul(),
//       new Transform.Rules.ReassociateXY(),
//       new Transform.Rules.XDivX(),
//     };
// }


// public sealed class TransposeDemoCase : FoldNopTransposeCase3
// {
//     public override Expr PreExpr
//     {
//         get
//         {
//             var input = torch.rand(1, 28, 28, 3).ToTensor();
//             var conv1 = NCHWToNHWC(DummyOp.Conv2D(NHWCToNCHW(input), 3, out_channels: 8, 3, 2));
//             var lhs = NCHWToNHWC(DummyOp.Conv2D(NHWCToNCHW(conv1), 8, out_channels: 8, 3, 1));
//             var rhs = conv1 + torch.rand(new long[] { 1, 14, 14, 8 }).ToTensor();
//             return lhs + rhs;
//         }
//     }
// }

