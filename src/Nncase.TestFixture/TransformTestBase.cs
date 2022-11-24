using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Transform;
using Nncase.Transform.Rules.Neutral;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
using Random = Nncase.IR.F.Random;

namespace Nncase.TestFixture;

// impl mixin by inherit interface with method had been impl
public partial class TransformTestBase : UnitTestFixtrue
{

    public override CompileOptions GetCompileOptions([CallerMemberName] string member_name = "")
    {
        var compileOptions = base.GetCompileOptions(member_name);
        compileOptions.QuantMode = QuantMode.UnsignedMode;
        compileOptions.QuantType = DataTypes.Int8;
        return compileOptions;
    }

    public override RunPassOptions GetPassOptions([CallerMemberName] string member_name = "")
    {
        var passOptions = base.GetPassOptions(member_name);
        passOptions = passOptions.SetRewriteOnce(true);
        return passOptions;
    }

    public virtual Expr TestMatched<T>(Expr pre, RunPassOptions passOptions) where T : IRewriteRule, new()
    {
        return TestMatchedCore(pre, passOptions, new T());
    }

    public void CondMatch<T>(bool cond, Expr expr, RunPassOptions passOptions) where T : IRewriteRule, new()
    {
        if (cond)
        {
            TestMatched<T>(expr, passOptions);
        }
        else
        {
            TestNotMatch<T>(expr, passOptions);
        }
    }

    public Expr TestMatchedCore(Expr pre, RunPassOptions passOptions, params IRewriteRule[] rules)
    {
        Assert.True(pre.InferenceType(), "TestInferFailed:" + pre.CheckedType);
        if (rules.Length == 0)
        {
            throw new InvalidOperationException("Rules should not be empty");
        }

        var post = CompilerServices.Rewrite(pre, rules, passOptions);
        Assert.NotEqual(pre, post);
        var v1 = pre.Evaluate();
        var v2 = post.Evaluate();

        Comparator.Compare(v1, v2);
        return post;
    }

    public void TestNotMatch(Expr pre, RunPassOptions passOptions, params IRewriteRule[] rules)
    {
        pre.InferenceType();
        var post = CompilerServices.Rewrite(pre, rules, passOptions);
        Assert.Equal(pre, post);
    }

    public void TestNotMatch<T>(Expr pre, RunPassOptions passOptions) where T : IRewriteRule, new()
    {
        TestNotMatch(pre, passOptions, new T());
    }

    // public void TestSwappableBinary<T>(BinaryOp op, Expr lhs, Expr rhs) where T : IRewriteRule, new()
    // {
    //     TestMatched<T>(Binary(op, lhs, rhs));
    //     TestMatched<T>(Binary(op, rhs, lhs));
    // }

    public Expr RewriteOnceFalse(Func<RunPassOptions, Expr> f, RunPassOptions passOptions)
    {
        var result = f(passOptions.SetRewriteOnce(false));
        return result;
    }

    public Expr Rewrite<T>(Expr pre, RunPassOptions passOptions) where T : IRewriteRule, new()
    {
        return CompilerServices.Rewrite(pre, new IRewriteRule[] { new T() }, passOptions);
    }

    public Expr RewriteWithSeq(Expr expr, RunPassOptions passOptions, IEnumerable<IRewriteRule> rules) =>
        rules.Aggregate(expr, (expr1, rule) => CompilerServices.Rewrite(expr1, new[] { rule }, passOptions));

    public Expr RewriteWithSeq(Expr expr, RunPassOptions passOptions, IEnumerable<IRewriteRule> lower,
        IEnumerable<IRewriteRule> folds, IEnumerable<IRewriteRule> fuse)
    {
        var l = RewriteWithSeq(expr, passOptions, lower);
        var s = RewriteOnceFalse((RunPassOptions opt) => RewriteWithSeq(l, opt, folds), passOptions);
        var f = RewriteWithSeq(s, passOptions, fuse);
        return f;
    }

    public Expr FoldNop(Expr expr, RunPassOptions passOptions) => RewriteOnceFalse((RunPassOptions opt)
      => CompilerServices.Rewrite(expr, new IRewriteRule[]
      {
          new FoldNopCast(),
          new FoldNopReshape()
      }, opt), passOptions);

    public Expr RewriteWithSeq(Expr expr, RunPassOptions passOptions, IEnumerable<IRewriteRule> lower,
        IRewriteRule fold, IEnumerable<IRewriteRule> fuse) => RewriteWithSeq(expr, passOptions, lower, new[] { fold }, fuse);

    public Expr RewriteWithSeq(Expr expr, RunPassOptions passOptions, IEnumerable<IRewriteRule> lower, IEnumerable<IRewriteRule> fuse) =>
        RewriteWithSeq(expr, passOptions, lower, new IRewriteRule[]
        {
            new FoldNopReshape(),
            new FoldNopCast()
        }, fuse);

    public Expr TestMultiMatched<T>(Expr expr, RunPassOptions passOptions, int count) where T : IRewriteRule, new() =>
        Enumerable.Range(0, count).Aggregate(expr, ((expr1, i) =>
        {
            var ex = TestMatched<T>(expr1, passOptions);
            return ex;
        }));

    public Expr RewriteMultiTimes<T>(Expr expr, RunPassOptions passOptions, int count) where T : IRewriteRule, new() =>
        Enumerable.Range(0, count).Aggregate(expr, ((expr1, i) =>
        {
            var ex = Rewrite<T>(expr1, passOptions);
            return ex;
        }));
}