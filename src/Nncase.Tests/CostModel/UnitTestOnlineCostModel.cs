// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.CostModel;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CostModelTest;

internal sealed class OnlineCostEvaluateVisitor : ExprVisitor<Cost, Unit>
{
    public OnlineCostEvaluateVisitor(string serverUrl)
    {
        ServerUrl = serverUrl;
        CompileModuleCallBack = DefaultCompileModule;
    }

    public OnlineCostEvaluateVisitor(string serverUrl, CompileModuleFunc compileModuleFunc) : this(serverUrl)
    {
        CompileModuleCallBack = compileModuleFunc;
    }

    public string ServerUrl { get; }

    public delegate string CompileModuleFunc(IRModule module);

    public CompileModuleFunc CompileModuleCallBack { get; }

    public bool IsServerOnline()
    {
        var client = new HttpClient();
        try
        {
            var response = client.SendAsync(new HttpRequestMessage(HttpMethod.Head, $"http://{ServerUrl}")).Result;

            // URL is online if we get a 2xx response status
            if (response.StatusCode != System.Net.HttpStatusCode.OK)
            {
                return false;
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    protected override Cost DefaultVisitLeaf(Expr expr)
    {
        return Cost.Zero;
    }

    protected override Cost VisitLeafCall(Call call)
    {
        var newArgs = new List<Expr>();
        var newVars = new List<Var>();
        var newArgPaths = new List<string>();
        for (int i = 0; i < call.Arguments.Length; i++)
        {
            if (call.Arguments[i] is not Const c)
            {
                var (tmpVar, path) = GetInput(i, call.Arguments[i].CheckedType);
                newVars.Add(tmpVar);
                newArgs.Add(tmpVar);
                newArgPaths.Add(path);
            }
            else
            {
                newArgs.Add(c);
            }
        }

        var newFunc = new Function("main", new Call(call.Target, newArgs.ToArray()), newVars.ToArray());
        var newModule = new IRModule(newFunc);
        var kmodelPath = CompileModuleCallBack(newModule);

        var uploadFiles = new[] { kmodelPath }.Concat(newArgPaths).ToArray();
        var time = SimulateKModel(uploadFiles);

        foreach (var item in uploadFiles)
        {
            File.Delete(item);
        }

        var cost = Cost.Zero;
        if (time == -1)
        {
            cost[CostFactorNames.CPUCycles] = UInt128.MaxValue;
        }
        else
        {
            cost[CostFactorNames.CPUCycles] = (UInt128)(time * 1.6 * 1e9);
        }

        return cost;
    }

    private static string DefaultCompileModule(IRModule module)
    {
        throw new NotImplementedException();
    }

    private float SimulateKModel(params string[] filePaths)
    {
        var client = new HttpClient();
        var formData = new MultipartFormDataContent();
        foreach (var (filePath, i) in filePaths.Select((s, i) => (s, i)))
        {
            var streamContent = new StreamContent(File.OpenRead(filePath));
            string fileName = i switch
            {
                0 => "kmodel",
                _ => $"input_{i}",
            };
            formData.Add(streamContent, "files", fileName);
        }

        var response = client.PostAsync($"http://{ServerUrl}/run_kmodel", formData).Result;
        var responseContent = response.Content.ReadAsStringAsync().Result;
        if (response.StatusCode != HttpStatusCode.OK)
        {
            return -1;
        }
        var time = float.Parse(responseContent);
        return time < 0.0 ? -1 : time;
    }

    private (Var var, string path) GetInput(int index, IRType type)
    {
        var inputVar = new Var($"tmpVar_{index}", type);
        if (type is TensorType tensorType)
        {
            var tensor = Nncase.IR.F.Random.Normal(tensorType.DType, 0, 1, 0, tensorType.Shape).Evaluate().AsTensor();
            string tempFilePath = Path.GetTempFileName();
            using (FileStream fs = File.OpenWrite(tempFilePath))
            {
                fs.Write(tensor.BytesBuffer);
            }

            return (inputVar, tempFilePath);
        }
        else
        {
            throw new NotSupportedException();
        }
    }
}




[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestOnlineCostModel : TestClassBase
{
    const string URL = "127.0.0.1:5000";
    [Fact]
    public void TestIsOnline()
    {
        var evaluator = new OnlineCostEvaluateVisitor(URL);

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

        Expr expr;
        {
            var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1, 224, 224, 3 }));
            expr = Nncase.IR.F.Tensors.Transpose(input, new[] { 0, 3, 1, 2 });
        }

        expr.InferenceType();

        var evaluator = new OnlineCostEvaluateVisitor(URL, (IRModule module) =>
        {
            var childContainer = CompilerServices.CreateScope();
            var childOptions = CompileOptions with { DumpFlags = Diagnostics.DumpFlags.None };
            var target = CompilerServices.CreateScope();
            childContainer.RegisterInstance(CompileSession.Target);
            childContainer.RegisterInstance(childOptions);
            var childSession = new CompileSession(childContainer, CompileSession.Target, childOptions);
            childContainer.RegisterInstance(childSession);

            var compiler = childSession.Compiler;
            compiler.ImportIRModule(module);
            compiler.CompileAsync().Wait();
            string tempFilePath = Path.GetTempFileName();
            using (FileStream fs = File.OpenWrite(tempFilePath))
            {
                compiler.Gencode(fs);
            }

            return tempFilePath;
        });

        Assert.True(evaluator.IsServerOnline());

        evaluator.Visit(expr);

        Assert.NotEqual(UInt128.MaxValue, evaluator.ExprMemo[expr].Score);
    }
}