// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for BufferOf.
/// </summary>
[TypeInferGenerator]
public partial class MatchBufferEvaluator : ITypeInferencer<MatchBuffer>, IOpPrinter<MatchBuffer>
{
    public string Visit(IIRPrinterContext context, MatchBuffer target, bool iLmode)
    {
        if (iLmode)
        {
            throw new System.NotSupportedException();
        }

        return $"Matched {context.GetArgument(target, MatchBuffer.Input)}";
    }

    private IRType Visit()
    {
        return TupleType.Void;
    }
}
