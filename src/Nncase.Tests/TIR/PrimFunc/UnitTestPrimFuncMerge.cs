// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Xunit;

namespace Nncase.Tests.TIRTest.PrimFuncTest;

[TestFixture.AutoSetupTestMethod(InitSession = true)]
public class UnitTestPrimFuncMerge : TestClassBase
{
    public static readonly TheoryData<IDataFlowPrimFuncCase, int> Datas = new()
    {
        { new DataFlowType10FusionCase(false), 18 },
        { new DataFlowType10FusionCase(true), 17 },
        { new DataFlowType9FusionCase(false), 16 },
        { new DataFlowType9FusionCase(true), 15 },
        { new DataFlowType8FusionCase(false), 14 },
        { new DataFlowType8FusionCase(true), 13 },
        { new DataFlowType7FusionCase(false), 12 },
        { new DataFlowType7FusionCase(true), 11 },
        { new DataFlowType6FusionCase(false), 10 },
        { new DataFlowType6FusionCase(true), 9 },
        { new DataFlowType5FusionCase(true), 8 },
        { new DataFlowType5FusionCase(false), 7 },
        { new DataFlowType4FusionCase(), 6 },
        { new DataFlowType3FusionCase(), 5 },
        { new DataFlowType2FusionCase(true), 4 },
        { new DataFlowType2FusionCase(false), 3 },
        { new DataFlowType1FusionCase(true), 2 },
        { new DataFlowType1FusionCase(false), 1 },
        { new DataFlowType0FusionCase(), 0 },
    };

    public IAnalyzerManager AnalyzerMananger => CompileSession.GetRequiredService<IAnalyzerManager>();

    [Theory]
    [MemberData(nameof(Datas))]
    private async void RunCore(IDataFlowPrimFuncCase fusionCase, int count)
    {
        var inputVar = new Var("input", new TensorType(DataTypes.Float32, PrimFuncBuilder.Dimensions));
        var main = new Function(fusionCase.BuildBody(inputVar), inputVar);

        CompilerServices.InferenceType(main);
#if DEBUG
        Dumpper.DumpDotIR(main, $"{count}_pre");
#endif
        var feedDict = new Dictionary<Var, IValue>(ReferenceEqualityComparer.Instance) {
          { inputVar, IR.F.Random.Normal(DataTypes.Float32, 0, 1, 12, PrimFuncBuilder.Dimensions).Evaluate() },
        };

        var exceptValue = new TestEvaluateVisitor(feedDict).Visit(main.Body);

        IRModule module = new(main);
        var pmgr = CompileSession.CreatePassManager("pmgr");
        pmgr.Add<DDrBufferSchdeulePass>(true);
        module = await pmgr.RunAsync(module);

        var post = (Function)module.Entry!;

#if DEBUG
        Dumpper.DumpDotIR(post, $"{count}_post");
#endif

        var visitor = new TestVisitor();
        visitor.Visit(post.Body);
        Assert.Equal(fusionCase.FinalWrapperCount, visitor.ExprMemo.Keys.Count(e => e is PrimFunctionWrapper));

        var actulValue = new TestEvaluateVisitor(feedDict).Visit(post.Body);
        Assert.Equal(exceptValue, actulValue);
    }
}

internal sealed class TestEvaluateVisitor : ExprVisitor<IValue, Unit>
{
    private readonly IReadOnlyDictionary<Var, IValue> _feedDict;

    public TestEvaluateVisitor(IReadOnlyDictionary<Var, IValue> feedDict)
    {
        _feedDict = feedDict;
    }

    protected override IValue DefaultVisitLeaf(Expr expr) => Value.None;

    protected override IValue VisitLeafConst(Const expr) => Value.FromConst(expr);

    /// <inheritdoc/>
    protected override IValue VisitLeafVar(Var expr)
    {
        if (!_feedDict.TryGetValue(expr, out var value))
        {
            throw new ArgumentException($"Must Set Input For Var {expr.Name}!");
        }

        return value;
    }

    protected override IValue VisitLeafCall(Call expr)
    {
        return expr.Target switch
        {
            PrimFunctionWrapper wrapper => new PrimFuncEvaluateVisitor(wrapper, expr.Arguments.AsValueEnumerable().Select(e => ExprMemo[e]).ToArray()).Evaluate(),
            Op op => expr.Evaluate(), // note current only support expr can be direct evaluate.
            _ => throw new NotImplementedException(expr.Target.ToString()),
        };
    }
}

internal sealed class PrimFuncEvaluateVisitor
{
    private static readonly int _pool_size = 1 * 4 * 8 * 9 * 4 * 30;
    private readonly PrimFunctionWrapper _wrapper;
    private readonly IValue[] _args;
    private readonly Dictionary<TIR.MemoryLocation, byte[]> _poolMap = new() {
          { TIR.MemoryLocation.Input, new byte[_pool_size] },
          { TIR.MemoryLocation.L2Data, new byte[_pool_size] },
          { TIR.MemoryLocation.Data, new byte[_pool_size] },
          { TIR.MemoryLocation.Output, new byte[_pool_size] },
        };

    public PrimFuncEvaluateVisitor(PrimFunctionWrapper wrapper, params IValue[] args)
    {
        _wrapper = wrapper;
        _args = args;
    }

    public IValue Evaluate()
    {
        // 1. copy input into input pool
        foreach (var (arg, param) in _args.Zip(_wrapper.Target.Parameters[.._wrapper.ParametersCount].ToArray()))
        {
            Assert.Equal(param.Size, arg.AsTensor().BytesBuffer.Length);
            arg.AsTensor().BytesBuffer.CopyTo(_poolMap[param.MemLocation].AsSpan(param.Start));
        }

        // 2. start l2 computing
        foreach (var statement in _wrapper.Target.Body.Fields)
        {
            EvaluateStatement(statement);
        }

        // 3. return output buffer
        var tensors = new List<Tensor>();
        foreach (var outputParam in _wrapper.Target.Parameters[_wrapper.ParametersCount..])
        {
            tensors.Add(Tensor.FromBytes(outputParam.ElemType, GetBufferSpan(outputParam).ToArray(), outputParam.FixedDimensions.ToArray()));
        }

        return tensors.Count == 1 ? Value.FromTensor(tensors[0]) : Value.FromTensors(tensors.ToArray());
    }

    private void EvaluateStatement(Expr statement)
    {
        switch (statement)
        {
            case Call { Target: LoadT } c:
                {
                    var ddr_pp = GetBufferSpan(c[LoadT.DdrPp]);
                    var glb_pp = GetBufferSpan(c[LoadT.GlbPp]);
                    ddr_pp.CopyTo(glb_pp);
                    break;
                }

            case Call { Target: StoreT } c:
                {
                    var ddr_pp = GetBufferSpan(c[StoreT.DdrPp]);
                    var glb_pp = GetBufferSpan(c[StoreT.GlbPp]);
                    glb_pp.CopyTo(ddr_pp);
                    break;
                }

            case Call { Target: BinaryT { BinaryOp: { } binary } } c:
                {
                    var glb_lhs_pp = MemoryMarshal.Cast<byte, float>(GetBufferSpan(c[BinaryT.GlbLhsPp]));
                    var glb_rhs_pp = MemoryMarshal.Cast<byte, float>(GetBufferSpan(c[BinaryT.GlbRhsPp]));
                    var glb_out_pp = MemoryMarshal.Cast<byte, float>(GetBufferSpan(c[BinaryT.GlbOutPp]));
                    for (int i = 0; i < glb_lhs_pp.Length; i++)
                    {
                        glb_out_pp[i] = binary switch
                        {
                            BinaryOp.Add => glb_lhs_pp[i] + glb_rhs_pp[i],
                            BinaryOp.Sub => glb_lhs_pp[i] - glb_rhs_pp[i],
                            BinaryOp.Mul => glb_lhs_pp[i] * glb_rhs_pp[i],
                            BinaryOp.Div => glb_lhs_pp[i] / glb_rhs_pp[i],
                            BinaryOp.Min => System.MathF.Min(glb_lhs_pp[i], glb_rhs_pp[i]),
                            BinaryOp.Max => System.MathF.Max(glb_lhs_pp[i], glb_rhs_pp[i]),
                            _ => throw new NotSupportedException(nameof(binary)),
                        };
                    }

                    break;
                }

            default:
                throw new ArgumentOutOfRangeException(nameof(statement));
        }
    }

    private Span<byte> GetBufferSpan(Expr expr)
    {
        var buffer = Assert.IsType<TIR.PhysicalBuffer>(expr);
        return _poolMap[buffer.MemLocation].AsSpan(buffer.Start, buffer.Size);
    }
}
