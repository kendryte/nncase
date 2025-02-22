// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using LanguageExt;
using Nncase.IR;
using Onnx;
using F = Nncase.IR.F;

namespace Nncase.Importer;

public partial class OnnxGraphImporter
{
    private Expr VisitIf(in NodeProto op)
    {
        var cond = GetInputExpr(op, 0);
        var thenGraph = GetGraphAttribute(op, "then_branch");
        var elseGraph = GetGraphAttribute(op, "else_branch");

        var thenImportResult = CreateSubgraphImporter(thenGraph).Import();
        var elseImportResult = CreateSubgraphImporter(elseGraph).Import();

        var thenVars = CreateFullVarsList(thenImportResult.Inputs, elseImportResult.Inputs);
        var elseVars = CreateFullVarsList(elseImportResult.Inputs, thenImportResult.Inputs);
        var thenFunc = AddFunction(thenGraph.Name, thenVars, thenImportResult.Outputs);
        var elseFunc = AddFunction(elseGraph.Name, elseVars, elseImportResult.Outputs);

        var arguments = thenVars.Select(x => GetInputExpr(x.Name)).ToArray();
        return new If(cond, thenFunc, elseFunc, arguments);
    }

    private Var[] CreateFullVarsList(Var[] origins, Var[] others)
    {
        var results = new List<Var>(origins);
        foreach (var other in others)
        {
            if (!origins.Any(x => x.Name == other.Name))
            {
                results.Add(other.With());
            }
        }

        results.Sort((x, y) => string.Compare(x.Name, y.Name, StringComparison.Ordinal));
        return results.ToArray();
    }
}
