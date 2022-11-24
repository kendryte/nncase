using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
public partial class TransformTestBase
{
    protected RunPassOptions passOptions;
    protected CompileOptions compileOptions;
    public TransformTestBase()
    {
        compileOptions = new CompileOptions();
        compileOptions.QuantMode = QuantMode.UnsignedMode;
        compileOptions.QuantType = DataTypes.Int8;
        passOptions = new RunPassOptions(CompilerServices.GetTarget(CompilerServices.CompileOptions.Target), 3, Testing.GetDumpDirPath(this.GetType()), compileOptions);
        passOptions = passOptions.SetRewriteOnce(true);
    }

    public virtual Expr TestMatched<T>(Expr pre) where T : IRewriteRule, new()
    {
        return TestMatchedCore(pre, new T());
    }

    public void CondMatch<T>(bool cond, Expr expr) where T : IRewriteRule, new()
    {
        if (cond)
        {
            TestMatched<T>(expr);
        }
        else
        {
            TestNotMatch<T>(expr);
        }
    }

    public Expr TestMatchedCore(Expr pre, params IRewriteRule[] rules)
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

    public void TestNotMatch(Expr pre, params IRewriteRule[] rules)
    {
        pre.InferenceType();
        var post = CompilerServices.Rewrite(pre, rules, passOptions);
        Assert.Equal(pre, post);
    }

    public void TestNotMatch<T>(Expr pre) where T : IRewriteRule, new()
    {
        TestNotMatch(pre, new T());
    }

    public void TestSwappableBinary<T>(BinaryOp op, Expr lhs, Expr rhs) where T : IRewriteRule, new()
    {
        TestMatched<T>(Binary(op, lhs, rhs));
        TestMatched<T>(Binary(op, rhs, lhs));
    }

    public Expr RewriteOnceFalse(Func<Expr> f)
    {
        passOptions = passOptions.SetRewriteOnce(false);
        var result = f();
        passOptions = passOptions.SetRewriteOnce(true);
        return result;
    }

    public Expr Rewrite(Expr pre, IEnumerable<IRewriteRule> rules)
    {
        return CompilerServices.Rewrite(pre, rules, passOptions);
    }

    public Expr Rewrite<T>(Expr pre) where T : IRewriteRule, new()
    {
        return Rewrite(pre, new IRewriteRule[] { new T() });
    }

    public Expr RewriteWithSeq(Expr expr, IEnumerable<IRewriteRule> rules) =>
        rules.Aggregate(expr, (expr1, rule) => Rewrite(expr1, new[] { rule }));

    public Expr RewriteWithSeq(Expr expr, IEnumerable<IRewriteRule> lower,
        IEnumerable<IRewriteRule> folds, IEnumerable<IRewriteRule> fuse)
    {
        var l = RewriteWithSeq(expr, lower);
        var s = RewriteOnceFalse(() => RewriteWithSeq(l, folds));
        var f = RewriteWithSeq(s, fuse);
        return f;
    }

    public Expr FoldNop(Expr expr) => RewriteOnceFalse(() => Rewrite(expr, new IRewriteRule[]
    {
        new FoldNopCast(),
        new FoldNopReshape()
    }));

    public Expr RewriteWithSeq(Expr expr, IEnumerable<IRewriteRule> lower,
        IRewriteRule fold, IEnumerable<IRewriteRule> fuse) => RewriteWithSeq(expr, lower, new[] { fold }, fuse);

    public Expr RewriteWithSeq(Expr expr, IEnumerable<IRewriteRule> lower, IEnumerable<IRewriteRule> fuse) =>
        RewriteWithSeq(expr, lower, new IRewriteRule[]
        {
            new FoldNopReshape(),
            new FoldNopCast()
        }, fuse);

    public Expr TestMultiMatched<T>(Expr expr, int count) where T : IRewriteRule, new() =>
        Enumerable.Range(0, count).Aggregate(expr, ((expr1, i) =>
        {
            var ex = TestMatched<T>(expr1);
            return ex;
        }));

    public Expr RewriteMultiTimes<T>(Expr expr, int count) where T : IRewriteRule, new() =>
        Enumerable.Range(0, count).Aggregate(expr, ((expr1, i) =>
        {
            var ex = Rewrite<T>(expr1);
            return ex;
        }));
}