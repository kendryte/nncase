// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CostModelTest;

// internal sealed class OnlineCostEvaluateVisitor : ExprVisitor<Cost, Unit>
// {
//     private readonly OnlineEGraphExtractCostEvaluator _evaluator;

//     public OnlineCostEvaluateVisitor(OnlineEGraphExtractCostEvaluator evaluator)
//     {
//         _evaluator = evaluator;
//     }

//     protected override Cost DefaultVisitLeaf(Expr expr)
//     {
//         return Cost.Zero;
//     }

//     protected override Cost VisitLeafCall(Call call)
//     {
//         return call.Target switch
//         {
//             Op op => _evaluator.Visit(new CostEvaluateContext(call), op),
//             _ => throw new NotSupportedException()
//         };
//     }

//     private sealed class CostEvaluateContext : ICostEvaluateContext
//     {
//         private readonly Call _currentCall;

//         public CostEvaluateContext(Call currentCall)
//         {
//             _currentCall = currentCall;
//         }

//         public T GetArgumentType<T>(Op op, ParameterInfo parameter) where T : IRType
//         {
//             return (T)_currentCall[parameter].CheckedType;
//         }

//         public T GetReturnType<T>() where T : IRType
//         {
//             return (T)_currentCall.CheckedType;
//         }

//         public bool TryGetConstArgument(Op op, ParameterInfo parameter, [MaybeNullWhen(false)] out Const @const)
//         {
//             bool ret = false;
//             @const = null;
//             if (_currentCall[parameter] is Const c)
//             {
//                 ret = true;
//                 @const = c;
//             }

//             return ret;
//         }
//     }
// }


[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestOnlineCostModel : TestClassBase
{
    const string URL = "127.0.0.1:5000";

    [Fact]
    public void TestIsOnline()
    {
        var evaluator = ActivatorUtilities.CreateInstance<OnlineCostEvaluateProvider>(CompileSession, URL);

        using (var server = new SimulatorServer(URL))
        {
            Assert.True(evaluator.IsServerOnline());
        }

        // Close the listener
        Assert.False(evaluator.IsServerOnline());
    }

    [Fact]
    public void TestRunKModel()
    {
        var server = new SimulatorServer(URL);

        Call expr;
        {
            var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 224, 224, 3 }));
            expr = Nncase.IR.F.Tensors.Transpose(input, new[] { 0, 3, 1, 2 });
        }

        expr.InferenceType();

        // var container = (IContainer)(IServiceProvider)CompileSession!;
        // container.Register<ICostEvaluateProvider, OnlineCostEvaluateProvider>(made: Parameters.Of.Type<string>(_ => URL));
        // var evaluator = new OnlineEGraphExtractCostEvaluator(URL);

        // CompilerServices.EvaluateOp


        // Assert.True(evaluator.IsServerOnline());

        // var visitor = new OnlineCostEvaluateVisitor(evaluator);
        // visitor.Visit(expr);

        // Assert.NotEqual(UInt128.MaxValue, visitor.ExprMemo[expr].Score);
    }
}