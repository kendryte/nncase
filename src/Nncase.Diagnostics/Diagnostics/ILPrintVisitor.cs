// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.IR.Math;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Diagnostics;

internal sealed class IndentedWriter : StreamWriter
{
    public IndentedWriter(Stream stream, bool leaveOpen = true)
        : base(stream, leaveOpen: leaveOpen)
    {
    }

    public int Indent { get; set; }

    public IndentedWriter WInd()
    {
        for (int i = 0; i < Indent; i++)
        {
            Write(' ');
        }

        return this;
    }
}

internal sealed class ScopeManager : IDisposable
{
    public ScopeManager()
    {
    }

    public event Action? ExitActions;

    public void Dispose()
    {
        ExitActions?.Invoke();
    }
}

internal sealed record ILPrintSymbol(string Name) : IPrintSymbol
{
    public string Span => throw new NotImplementedException();

    public bool IsRefSymobl => true;

    public override string ToString() => Name;
}

internal sealed class PrintOpContext : IPrintOpContext
{
    public PrintOpContext(PrinterFlags flags, ILPrintSymbol op, ILPrintSymbol[] arguments)
    {
        Flags = flags;
        Op = op;
        Arguments = arguments;
    }

    public PrinterFlags Flags { get; }

    public ILPrintSymbol Op { get; }

    public ILPrintSymbol[] Arguments { get; }

    public IPrintSymbol Get(Op op) => Op;

    public IPrintSymbol GetArgument(Op op, ParameterInfo parameter) => Arguments[parameter.Index];

    public IPrintSymbol[] GetArguments(Op op) => Arguments;

    public string Indent() => string.Empty;

    public IPrintSymbol Visit(Expr expr) => throw new NotImplementedException();
}

internal sealed class ILPrintVisitor : ExprFunctor<string, string>
{
    private readonly IndentedWriter _writer;
    private readonly List<Dictionary<Expr, string>> _stackedMemos;
    private readonly List<int> _stackedSSANumbers;
    private readonly List<int> _stackedScopeDepthOffSets;
    private readonly List<int> _stackedVisitDepthOffSets;

    public ILPrintVisitor(IndentedWriter printer, PrinterFlags printerFlags, IReadOnlyDictionary<Expr, string> feedDict)
    {
        _writer = printer;
        Flags = printerFlags;
        _stackedMemos = new() { new(feedDict, ReferenceEqualityComparer.Instance) };
        _stackedSSANumbers = [0];
        _stackedScopeDepthOffSets = [0];
        _stackedVisitDepthOffSets = [0];
    }

    public PrinterFlags Flags { get; private set; }

    public int ScopeDepth => _stackedMemos.Count;

    public int VisitDepth { get; private set; }

    public override string DefaultVisitType(IRType type) => type.ToString();

    /// <inheritdoc/>
    public override string VisitType(AnyType type) => "any";

    /// <inheritdoc/>
    public override string VisitType(CallableType type) =>
        $"({string.Join(", ", type.Parameters.Select(VisitType))}) -> {VisitType(type.ReturnType)}";

    /// <inheritdoc/>
    public override string VisitType(InvalidType type) => $"invalid:{type.Reason}";

    /// <inheritdoc/>
    public override string VisitType(NoneType type) => $"";

    /// <inheritdoc/>
    public override string VisitType(TensorType type) => type.DType switch
    {
        PrimType ptype => ptype.GetDisplayName() + (type.Shape.IsScalar ? string.Empty : VisitShape(type.Shape)),
        PointerType { ElemType: PrimType etype } => $"*{etype.GetDisplayName()}",
        ValueType => $"{type.DType}",
        VectorType vtype => $"{vtype.ElemType.GetDisplayName()}<{string.Join(",", vtype.Lanes)}>" + (type.Shape.IsScalar ? string.Empty : VisitShape(type.Shape)),
        _ => throw new NotSupportedException(type.DType.GetType().Name),
    };

    /// <inheritdoc/>
    public override string VisitType(TupleType type) =>
        $"({string.Join(", ", type.Fields.Select(VisitType))})";

    /// <inheritdoc/>
    public override string VisitType(DistributedType type)
    {
        var shape = type.TensorType.Shape.ToArray();
        foreach (var (s, r) in type.AxisPolices.Select((s, r) => (s, r)))
        {
            if (s is SBPSplit split)
            {
                if (shape[r].IsFixed)
                {
                    shape[r] = shape[r] / split.Axes.Select(a => type.Placement.Hierarchy[a]).Aggregate(1, (a, b) => a * b);
                }
            }
        }

        var sshape = shape.Select(s => s.ToString()).ToArray();
        foreach (var (s, r) in type.AxisPolices.Select((s, r) => (s, r)))
        {
            if (s is SBPSplit split)
            {
                sshape[r] += string.Join(string.Empty, split.Axes.Select(a => $"@{type.Placement.Name[a]}"));
            }
        }

        return $"{{{VisitType(type.TensorType)}, ({string.Join(',', type.AxisPolices)}), [{string.Join(',', sshape)}]}}";
    }

    protected override string DispatchVisit(Expr expr)
    {
        if (_stackedMemos[^1].TryGetValue(expr, out var name))
        {
            return name;
        }

        VisitDepth++;
        if (ShouldEnterVisit())
        {
            name = base.DispatchVisit(expr);
        }
        else
        {
            name = $"...";
        }

        VisitDepth--;
        _stackedMemos[^1].Add(expr, name);
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitCall(Call expr)
    {
        var target = Visit(expr.Target);
        var args = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        var context = new PrintOpContext(Flags, new(target), args.Select(arg => new ILPrintSymbol(arg)).ToArray());
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            string? callOpString = null;
            if (expr.Target is Op op)
            {
                callOpString = CompilerServices.PrintOp(op, context);
            }

            callOpString ??= $"{target}({string.Join(", ", args)})";
            return callOpString;
        }
        else
        {
            var name = GetNextSSANumber();
            string property = string.Empty;
            if (expr.Target is Op op)
            {
                property = op.DisplayProperty() is string prop && prop != string.Empty ? (prop + ", ") : string.Empty;
            }

            _writer.WInd().Write($"{name} = {target}({property}{string.Join(", ", args)})");
            AppendCheckedType(expr.CheckedType, expr.Metadata.Range, " " + string.Join(",", expr.Metadata.OutputNames ?? Array.Empty<string>()));
            return name;
        }
    }

    /// <inheritdoc/>
    protected override string VisitIf(If expr)
    {
        var cond = Visit(expr.Condition);
        var thenFunc = Visit(expr.Then);
        var elseFunc = Visit(expr.Else);
        var args = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            return $"({cond} ? {thenFunc}({string.Join(", ", args)}) : {elseFunc}({string.Join(", ", args)}))";
        }
        else
        {
            var name = GetNextSSANumber();
            _writer.WInd().Write($"{name} = if({cond}, {string.Join(", ", args)})");
            AppendCheckedType(expr.CheckedType, expr.Metadata.Range);
            _writer.WInd().WriteLine("{");
            using (IndentScope())
            {
                _writer.WInd().WriteLine(thenFunc);
            }

            _writer.WInd().WriteLine("} else {");

            using (IndentScope())
            {
                _writer.WInd().WriteLine(elseFunc);
            }

            _writer.WInd().WriteLine("}");
            return name;
        }
    }

    /// <inheritdoc/>
    protected override string VisitConst(Const expr)
    {
        string valueStr = expr switch
        {
            TensorConst tc => VisitTensorValue(tc.Value, tc.ValueType),
            TupleConst tp => VisitValue(tp.Value),
            _ => throw new ArgumentOutOfRangeException(nameof(expr)),
        };

        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            return valueStr;
        }

        valueStr = valueStr != string.Empty ? " : " + valueStr : string.Empty;
        var name = $"const({VisitType(expr.CheckedType)}{valueStr})";
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitFunction(Function expr)
    {
        using var subScope = NestedScope();
        var name = $"%{expr.Name}";
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            return name;
        }

        // 1. Function signature
        _writer.WInd().Write($"{name} = fn<{expr.ModuleKind}>({StringUtility.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit))})");
        AppendCheckedType(expr.CheckedType, expr.Metadata.Range);
        _writer.WInd().WriteLine("{");

        // 2. Function body
        if (ShouldEnterScope())
        {
            using (IndentScope())
            {
                var body = Visit(expr.Body);
            }
        }
        else
        {
            _writer.WInd().WriteLine("...");
        }

        // 3. Function closing
        _writer.WInd().WriteLine("}");
        return name;
    }

    protected override string VisitFusion(Fusion expr)
    {
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            return expr.Name;
        }
        else
        {
            using var subScope = NestedScope();
            var name = $"%{expr.Name}";

            // 1. Function signature
            _writer.WInd().Write($"{name} = fusion<{expr.ModuleKind}>({StringUtility.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit))})");
            AppendCheckedType(expr.CheckedType, expr.Metadata.Range);
            _writer.WInd().WriteLine("{");

            // 2. Function body
            if (ShouldEnterScope())
            {
                using (IndentScope())
                {
                    Visit(expr.Body);
                }
            }
            else
            {
                _writer.WInd().WriteLine("...");
            }

            // 3. Function closing
            _writer.WInd().WriteLine("}");
            return name;
        }
    }

    /// <inheritdoc/>
    protected override string VisitPrimFunctionWrapper(PrimFunctionWrapper expr)
    {
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            return expr.Name;
        }

        using var subScope = NestedScope();
        var name = $"%{expr.Name}";

        // 1. Function signature
        _writer.WInd().Write($"{name} = prim_wrapper({string.Join(", ", expr.ParameterTypes.Select(x => x == null ? string.Empty : VisitType(x)))})");
        AppendCheckedType(expr.CheckedType, expr.Metadata.Range, " {");

        // 2. Function body
        if (ShouldEnterScope())
        {
            using (IndentScope())
            {
                using (var bodys = new StringReader(CompilerServices.Print(expr.Target, Flags | PrinterFlags.Script)))
                {
                    while (bodys.ReadLine() is string line)
                    {
                        _writer.WInd().WriteLine(line);
                    }
                }
            }
        }
        else
        {
            _writer.WInd().WriteLine("...");
        }

        // 3. Function closing
        _writer.WInd().WriteLine("}");
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitOp(Op expr)
    {
        return expr.GetType().Name;
    }

    /// <inheritdoc/>
    protected override string VisitTuple(IR.Tuple expr)
    {
        var fields = expr.Fields.AsValueEnumerable().Select(Visit).ToArray();
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            return $"({string.Join(", ", fields)})";
        }
        else
        {
            var name = GetNextSSANumber();
            _writer.WInd().Write($"{name} = ({string.Join(", ", fields)})");
            AppendCheckedType(expr.CheckedType, expr.Metadata.Range);
            _writer.WInd().WriteLine();
            return name;
        }
    }

    /// <inheritdoc/>
    protected override string VisitVar(Var expr)
    {
        var name = $"%{expr.Name}#{expr.GlobalVarIndex}";
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            return name;
        }

        name += $": {VisitType(expr.TypeAnnotation)}";
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitNone(None expr)
    {
        return $"None";
    }

    /// <inheritdoc/>
    protected override string VisitMarker(Marker expr)
    {
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            throw new NotSupportedException($"Inline Mode with {typeof(Marker)}");
        }

        var name = GetNextSSANumber();
        var target = Visit(expr.Target);
        var attr = Visit(expr.Attribute);
        _writer.WInd().Write($"{name} = {target}@({expr.Name} = {attr})");
        AppendCheckedType(expr.CheckedType, expr.Metadata.Range);
        return name;
    }

    protected override string VisitBuffer(TIR.Buffer expr)
    {
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            throw new NotSupportedException($"Inline Mode with {typeof(TIR.Buffer)}");
        }

        var name = GetNextSSANumber();
        var type = expr.DistributedType == null ? VisitType(expr.CheckedType) : VisitType(expr.DistributedType);
        _writer.WInd().WriteLine($"{name} = buffer({type})");
        return name;
    }

    protected override string VisitBufferOf(BufferOf expr)
    {
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            throw new NotSupportedException($"Inline Mode with {typeof(BufferOf)}");
        }

        var name = $"bufferof({Visit(expr.Input)})";
        return name;
    }

    /// <inheritdoc/>
    protected override string VisitGrid(IR.Affine.Grid expr)
    {
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            throw new NotSupportedException($"Inline Mode with {typeof(IR.Affine.Grid)}");
        }

        var name = GetNextSSANumber();
        var reads = expr.Reads.AsValueEnumerable().Select(Visit).ToArray();
        var buffers = expr.Buffers.AsValueEnumerable().Select(Visit).ToArray();

        // 1. For Loop signature
        _writer.Write($"{name} = Grid({string.Join(", ", reads)})");
        AppendCheckedType(expr.CheckedType, expr.Body.Metadata.Range);
        _writer.WInd().WriteLine(" {");

        using (IndentScope())
        {
            // 2. In buffers
            _writer.WInd().WriteLine($"Reads:");
            using (IndentScope())
            {
                for (int i = 0; i < buffers.Length - 1; i++)
                {
                    _writer.WInd().WriteLine($"{buffers[i]}: {expr.AccessMaps[i]}");
                }
            }

            // 3. Out buffer
            _writer.WInd().WriteLine($"Write:");
            using (IndentScope())
            {
                _writer.WInd().WriteLine($"{buffers[^1]}: {expr.AccessMaps[^1]}");
            }

            // 4. For Body
            var domain_parameters = Visit(expr.DomainParameter);
            var parameters = expr.BodyParameters.AsValueEnumerable().Select(Visit).ToArray();
            _writer.WInd().Write($"Body: ({domain_parameters}, {string.Join(", ", parameters)})");
            AppendCheckedType(expr.Body.CheckedType, expr.Body.Metadata.Range, " {", hasNewLine: true);
            using (IndentScope())
            {
                var ss = CompilerServices.Print(expr.Body, Flags | PrinterFlags.Script);
                foreach (var line in ss.Split('\n'))
                {
                    _writer.WInd().WriteLine(line);
                }
            }

            _writer.WInd().WriteLine("}");
        }

        // 3. For closing
        _writer.WInd().WriteLine("}");

        return name;
    }

    protected override string VisitShape(Shape shape) => shape.ToString();

    protected override string VisitDimension(Dimension expr) => expr.ToString();

    private string GetNextSSANumber()
    {
        return $"%{_stackedSSANumbers[^1]++}";
    }

    private void AppendCheckedType(IRType? type, ValueRange<double>? range, string end = "", bool hasNewLine = true)
    {
        var rangeText = range is not null ? $" [{range.Value.Min}, {range.Value.Max}]" : string.Empty;
        if (type is not null)
        {
            if (hasNewLine)
            {
                _writer.WriteLine($": // {VisitType(type)}{end}, {rangeText}");
            }
            else
            {
                _writer.WriteLine($": // {VisitType(type)}{end}, {rangeText}");
            }
        }
        else
        {
            _writer.WriteLine();
        }
    }

    private string VisitValue(IValue value)
    {
        return value switch
        {
            TensorValue tv => VisitTensorValue(tv.AsTensor()),
            TupleValue tp => $"({StringUtility.Join(",", tp.AsValueEnumerable().Select(VisitValue))})",
            _ => throw new NotSupportedException(nameof(value)),
        };
    }

    private string VisitTensorValue(Tensor tensor, IRType? valueType = null)
    {
        var length = 8;
        if (Flags.HasFlag(PrinterFlags.Normal))
        {
            length = 32;
        }
        else if (Flags.HasFlag(PrinterFlags.Detailed))
        {
            length = int.MaxValue;
        }

        if (tensor.Length <= length)
        {
            return tensor.GetArrayString(false);
        }

        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            return VisitType(valueType ?? new TensorType(tensor.ElementType, tensor.Shape));
        }

        return string.Empty;
    }

    private bool ShouldEnterVisit()
    {
        if (Flags.HasFlag(PrinterFlags.Inline))
        {
            if (Flags.HasFlag(PrinterFlags.Minimal) && (_stackedVisitDepthOffSets[^1] + VisitDepth) < 8)
            {
                return true;
            }
            else if (Flags.HasFlag(PrinterFlags.Normal) && (_stackedVisitDepthOffSets[^1] + VisitDepth) < 16)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        return true;
    }

    private bool ShouldEnterScope()
    {
        if (Flags.HasFlag(PrinterFlags.Minimal) && (_stackedScopeDepthOffSets[^1] + ScopeDepth) <= 2)
        {
            return true;
        }
        else if (Flags.HasFlag(PrinterFlags.Normal) && (_stackedScopeDepthOffSets[^1] + ScopeDepth) <= 4)
        {
            return true;
        }
        else if (Flags.HasFlag(PrinterFlags.Detailed))
        {
            return true;
        }

        return false;
    }

    private ScopeManager NestedScope(PrinterFlags? flags = null, int indentDiff = 0, int visitDepthDiff = 0, int scopeDepthDiff = 0)
    {
        var parentFlags = Flags;
        Flags = flags ?? parentFlags;
        _writer.Indent += indentDiff;
        _stackedSSANumbers.Add(_stackedSSANumbers[^1]);
        _stackedMemos.Add(new(_stackedMemos[^1], ReferenceEqualityComparer.Instance));
        _stackedScopeDepthOffSets.Add(scopeDepthDiff);
        _stackedVisitDepthOffSets.Add(visitDepthDiff);
        var manager = new ScopeManager();
        manager.ExitActions += () =>
        {
            _writer.Indent -= indentDiff;
            Flags = parentFlags;
            _stackedSSANumbers[^2] = _stackedSSANumbers[^1];
            var i = _stackedSSANumbers.Count - 1;
            _stackedSSANumbers.RemoveAt(i);
            _stackedMemos.RemoveAt(i);
            _stackedScopeDepthOffSets.RemoveAt(i);
            _stackedVisitDepthOffSets.RemoveAt(i);
        };
        return manager;
    }

    private ScopeManager IndentScope(int indentDiff = 2)
    {
        _writer.Indent += indentDiff;
        var manager = new ScopeManager();
        manager.ExitActions += () => _writer.Indent -= indentDiff;
        return manager;
    }
}
