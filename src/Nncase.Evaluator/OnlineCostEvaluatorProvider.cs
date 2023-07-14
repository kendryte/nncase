// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.CostModel;
using Nncase.IR;

[assembly: InternalsVisibleTo("Nncase.Tests")]

namespace Nncase.Evaluator;

internal sealed class OnlineCostEvaluateProvider : ICostEvaluateProvider
{
    private readonly Func<IRModule, string> _moduleCompiler;

    public OnlineCostEvaluateProvider(string serverUrl, Func<IRModule, string> moduleCompiler)
    {
        ServerUrl = serverUrl;
        _moduleCompiler = moduleCompiler;
    }

    public string ServerUrl { get; }

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

    public Cost EvaluateCost(Expr expr) => throw new NotImplementedException();

    public Cost EvaluateOpCost(Op op, ICostEvaluateContext context)
    {
        var newArgs = new List<Expr>();
        var newVars = new List<Var>();
        var newArgPaths = new List<string>();
        foreach (var parameter in op.Parameters)
        {
            if (context.TryGetConstArgument(op, parameter, out var c))
            {
                newArgs.Add(c);
            }
            else
            {
                var (tmpVar, path) = CreateInput(parameter.Index, context.GetArgumentType<IRType>(op, parameter));
                newVars.Add(tmpVar);
                newArgs.Add(tmpVar);
                newArgPaths.Add(path);
            }
        }

        var newFunc = new Function("tmp_main", new Call(op, newArgs.ToArray()), newVars.ToArray());
        var newModule = new IRModule(newFunc);
        return RunModule(newModule, newArgPaths);
    }

    public Cost EvaluateBaseFuncCost(BaseFunction baseFunction, ICostEvaluateContext context)
    {
        if (baseFunction is not Fusion fusion)
        {
            return Cost.Zero;
        }

        using var pinner = new ExprPinner(fusion);
        var newArgPaths = new List<string>();
        foreach (var (t, i) in baseFunction.ParameterTypes.Select((t, i) => (t, i)))
        {
            var (v, path) = CreateInput(i, t!);
            newArgPaths.Add(path);
        }

        var newfunc = new Function("tmp_main", fusion.Body, fusion.Parameters);
        var module = new IRModule(newfunc);
        return RunModule(module, newArgPaths);
    }

    private Cost RunModule(IRModule module, List<string> argPaths)
    {
        var kmodelPath = _moduleCompiler(module);
        var uploadFiles = new[] { kmodelPath }.Concat(argPaths).ToArray();
        var time = RunKModel(uploadFiles);
        foreach (var item in uploadFiles)
        {
            File.Delete(item);
        }

        var cost = new Cost();
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

    private float RunKModel(params string[] filePaths)
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

    private (Var Var, string Path) CreateInput(int index, IRType type)
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
