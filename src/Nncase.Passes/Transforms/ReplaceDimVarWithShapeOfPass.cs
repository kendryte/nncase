// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Passes.Rules;

namespace Nncase.Passes.Transforms;

/// <summary>
/// Shape inference.
/// </summary>
public sealed class ReplaceDimVarWithShapeOfPass : FunctionPass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ReplaceDimVarWithShapeOfPass"/> class.
    /// </summary>
    public ReplaceDimVarWithShapeOfPass()
    {
    }

    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction pre, RunPassContext options)
    {
        if (pre is Function func)
        {
            var varMap = CreateDimVarMap();
            var visitor = new ReplaceDimVarWithShapeOfVisitor(varMap);
            var newBody = visitor.Visit(func.Body, default);
            var post = (BaseFunction)func.With(body: newBody);
            return Task.FromResult(post);
        }

        return Task.FromResult(pre);
    }

    private Dictionary<DimVar, Dimension> CreateDimVarMap()
    {
        var shapeOfs = new Dictionary<Var, Expr>(ReferenceEqualityComparer.Instance);
        var varMap = new Dictionary<DimVar, Dimension>(ReferenceEqualityComparer.Instance);
        foreach (var (tensorVar, dimExprs) in CompileSession.CompileOptions.ShapeBucketOptions.VarMap)
        {
            for (int i = 0; i < dimExprs.Length; i++)
            {
                var dimExpr = dimExprs[i];
                if (dimExpr is DimVar dimVar)
                {
                    if (!varMap.ContainsKey(dimVar))
                    {
                        if (!shapeOfs.TryGetValue(tensorVar, out var shapeOf))
                        {
                            shapeOf = IR.F.Tensors.ShapeOf(tensorVar);
                            shapeOfs.Add(tensorVar, shapeOf);
                        }

                        varMap.Add(dimVar, shapeOf[i].AsDim());
                    }
                }
            }
        }

        return varMap;
    }
}

internal sealed class ReplaceDimVarWithShapeOfVisitor : ExprCloner<Unit>
{
    private readonly IReadOnlyDictionary<DimVar, Dimension> _varMap;

    public ReplaceDimVarWithShapeOfVisitor(IReadOnlyDictionary<DimVar, Dimension> varMap)
    {
        _varMap = varMap;
        CloneUnmutated = false;
    }

    protected override Expr VisitLeafDimVar(DimVar expr, Unit context)
    {
        if (_varMap.TryGetValue(expr, out var shapeOf))
        {
            return shapeOf;
        }

        return expr;
    }
}
