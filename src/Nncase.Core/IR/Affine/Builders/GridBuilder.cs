// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.TIR;

namespace Nncase.IR.Affine.Builders;

/// <summary>
/// builfer the grid.
/// </summary>
public interface IGridBuilder : IExprBuilder<Grid>
{
    /// <summary>
    /// else grid.
    /// </summary>
    /// <param name="exprOrBuilders"> statements. </param>
    /// <returns> GridBuilder. </returns>
    IGridBuilder Body(params object[] exprOrBuilders);

    IGridBuilder Read(Expr argument, AffineMap accessMap, out Var parameter);

    IGridBuilder Write(Expr buffer, AffineMap accessMap, out Var parameter);
}

internal class GridBuilder : IGridBuilder
{
    private readonly List<Var> _bodyParameters = new();
    private readonly List<object> _body = new();
    private readonly List<Expr> _reads = new();
    private readonly List<Expr> _readBuffers = new();
    private readonly List<AffineMap> _readMaps = new();
    private readonly IRType _returnType;
    private readonly string _moduleKind;
    private Expr? _writeBuffer;
    private AffineMap? _writeMap;

    public GridBuilder(IRType outputType, string moduleKind = Callable.StackVMModuleKind)
    {
        _returnType = outputType;
        _moduleKind = moduleKind;
    }

    public IGridBuilder Body(params object[] exprOrBuilders)
    {
        _body.AddRange(exprOrBuilders);
        return this;
    }

    public Grid Build()
    {
        return new Grid(
            _returnType,
            _moduleKind,
            CollectionsMarshal.AsSpan(_bodyParameters),
            _readMaps.Append(_writeMap ?? throw new InvalidOperationException("Write map is not set.")).ToArray(),
            _readBuffers.Append(_writeBuffer ?? throw new InvalidOperationException("Write buffer is not set.")).ToArray(),
            CollectionsMarshal.AsSpan(_reads),
            Sequential.Flatten(CollectionsMarshal.AsSpan(_body)));
    }

    public IGridBuilder Read(Expr argument, AffineMap accessMap, out Var parameter)
    {
        parameter = new Var(new TensorType(argument.CheckedDataType, Shape.Unknown(argument.CheckedShape.Rank)));
        _bodyParameters.Add(parameter);
        _reads.Add(argument);
        _readBuffers.Add(F.Buffer.BufferOf(argument));
        _readMaps.Add(accessMap);
        return this;
    }

    public IGridBuilder Write(Expr buffer, AffineMap accessMap, out Var parameter)
    {
        parameter = new Var(new TensorType(buffer.CheckedDataType, Shape.Unknown(buffer.CheckedShape.Rank)));
        _bodyParameters.Add(parameter);
        _writeBuffer = buffer;
        _writeMap = accessMap;
        return this;
    }
}
