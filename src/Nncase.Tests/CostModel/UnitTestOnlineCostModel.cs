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

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestOnlineCostModel : TestClassBase
{
    private const string URL = "127.0.0.1:5000";

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

        var evaluator = new OnlineCostEvaluateProvider(URL, m =>
        {
            var compiler = CompileSession.New<ICompiler>();
            compiler.ImportIRModule(m);
            var path = Path.GetTempFileName();
            using (var fs = File.OpenWrite(path))
            {
                compiler.Gencode(fs);
            }

            return path;
        });

        Assert.True(evaluator.IsServerOnline());
    }
}
