// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.Passes.Rules.ShapeBucket;

namespace Nncase.Tests;

public static class ShapeBucketTestHelper
{
    internal static IRModule MakeModule(Expr output, Var[] inputVar) => new(new Function("main", output, inputVar));

    internal static Call MakeSingleSimpleFusionCall(Func<Expr, Expr> ctor, Expr arg)
    {
        var v = new Var(arg.CheckedType);
        var abs = ctor(v);
        var f = new BucketFusion("stackvm", abs, new[] { v }, Array.Empty<Var>());
        var c = new Call(f, arg);
        return c;
    }

    internal static Call MakeSimpleFusionCall(Func<Expr[], Expr> ctor, params Expr[] args)
    {
        var paramList = args.Select(x => new Var(x.CheckedType)).ToArray();
        var abs = ctor(paramList);
        var f = new BucketFusion("stackvm", abs, paramList, Array.Empty<Var>());
        var c = new Call(f, args);
        return c;
    }
}
