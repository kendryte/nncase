// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator;

internal sealed class ShapeEvaluateContext : IShapeEvaluateContext
{
    private readonly Dictionary<Expr, Expr> _memo;
    public IReadOnlyDictionary<Var, Expr[]> VarMap { get; }
    public ShapeEvaluateContext(Dictionary<Expr, Expr> memo, IReadOnlyDictionary<Var, Expr[]> varMap)
    {
        _memo = memo;
        VarMap = varMap;
    }

    public Call? CurrentCall { get; set; }

    public Expr GetArgument(Op op, ParameterInfo parameter)
    {
        if (op.GetType() == parameter.OwnerType)
        {
            return GetCurrentCall().Arguments[parameter.Index];
        }
        else
        {
            throw new ArgumentOutOfRangeException($"Operator {op} doesn't have parameter: {parameter.Name}.");
        }
    }

    public Expr[] GetArguments(Op op, params ParameterInfo[] paramsInfo)
    {
        return paramsInfo.Select(info => GetArgument(op, info)).ToArray();
    }

    public Expr GetArgumentShape(Op op, ParameterInfo parameter)
    {
        // todo: try get, or ShapeOf
        return _memo[GetArgument(op, parameter)];
    }

    private Call GetCurrentCall() => CurrentCall ?? throw new InvalidOperationException("Current call is not set.");
}
