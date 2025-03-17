// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for BufferOf.
/// </summary>
[TypeInferGenerator]
public partial class MatchBufferEvaluator : ITypeInferencer<MatchBuffer>, IOpPrinter<MatchBuffer>
{
    public string Visit(IPrintOpContext context, MatchBuffer target)
    {
        if (context.Flags.HasFlag(PrinterFlags.Inline) || context.Flags.HasFlag(PrinterFlags.Script))
        {
            return $"Matched {context.GetArgument(target, MatchBuffer.Input)}";
        }

        return context.GetDefault(target);
    }

    private IRType Visit()
    {
        return TupleType.Void;
    }
}
