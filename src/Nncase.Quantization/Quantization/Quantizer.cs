// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using Google.OrTools.Graph;
using Microsoft.Extensions.Options;
using NetFabric.Hyperlinq;
using Newtonsoft.Json;
using Nncase.Diagnostics;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Quantization;
internal partial class Quantizer
{
    private readonly IEGraph _graph;
    private readonly Expr _expr;
    private readonly QuantizeOptions _quantizeOptions;
    private readonly Dictionary<Var, QuantConfig>? _fakeNodeConfigs = new();
    private readonly Dictionary<Var, Expr> _fakeNodesVars = new();

    public Quantizer(IEGraph graph, QuantizeOptions quantizeOptions)
    {
        _graph = graph;
        _expr = ExtractExpr();
        _quantizeOptions = quantizeOptions;

        var dumpPath = Path.Join(DumpScope.Current.Directory, "..", "..", "..", "/");
        EGraphPrinter.DumpEgraphAsDot(_graph, dumpPath + "_graph.dot");
        CompilerServices.DumpIR(_expr, "_expr", dumpPath);
    }

    public async Task RunAsync(RunPassContext options)
    {
        var ranges = new Dictionary<Expr, ValueRange<float>[]>(ReferenceEqualityComparer.Instance);
        var quantTypes = new Dictionary<Expr, Dictionary<ParameterInfo, DataType>>();
        List<KeyValuePair<(Var Var, QuantConfig QuantConfig), float>> quantSensitivity = new();
        Tensor groundTruth = default!;

        // 如果量化方式由json文件指定，那么直接从json文件中读取
        if (_quantizeOptions.QuantScheme != string.Empty)
        {
            UpdateQuantInfoFromJson();
        }
        else
        {
            if (_quantizeOptions.CalibrationDataset == null)
            {
                throw new InvalidOperationException($"{nameof(_quantizeOptions.CalibrationDataset)} is not set");
            }

            // 1.0 Get ranges
            ranges = (Dictionary<Expr, ValueRange<float>[]>)await GetRangesAsync(_quantizeOptions.CalibrationDataset!);

            // ranges = ranges.ToDictionary(item => item.Key, item => FixUpRange(item.Value));
            ranges = ranges.ToDictionary(item => item.Key, item => item.Value.Select(v => FixUpRange(v)).ToArray());

            if (_quantizeOptions.CalibrationMethod is CalibMethod.Kld)
            {
                int srcBinSize = 8192;
                int dstBinSize = 256;

                // 1.1. Get histograms
                var histograms = await GetHistogramsAsync(_quantizeOptions.CalibrationDataset, ranges, srcBinSize, dstBinSize);

                // 1.2. Select best ranges
                ranges = (Dictionary<Expr, ValueRange<float>[]>)GetOptRanges(histograms, ranges, srcBinSize, dstBinSize, _quantizeOptions.CalibrationMethod);
            }

            // 如果量化方式由config.toml文件指定，那么直接按照config.toml中指定的全局量化方式进行量化
            if (_quantizeOptions.SensitivityQuantEnabled == false)
            {
                UpdateQuantTypesByConfig(quantTypes, _quantizeOptions.QuantType, _quantizeOptions.WQuantType);
            }
            else
            {
                // 如果量化方式需要基于敏感度排序进行量化，则直接按照敏感度排序后的量化方式进行量化
                (quantSensitivity, groundTruth) = await GetSensitivity(ranges, quantTypes);
            }

            FillQuantConfig(ranges, quantTypes);

            await UpdateQuantConfigBySensitivity(quantSensitivity, groundTruth);
        }

        AssignRanges();

        if (_quantizeOptions.ExportQuantScheme == true)
        {
            ExportQuantScheme();
        }
    }

    private void UpdateQuantInfoFromJson()
    {
        string readJson = _quantizeOptions.QuantScheme;
        using (var r = new StreamReader(readJson))
        {
            string json = r.ReadToEnd();
            var quantScheme = JsonConvert.DeserializeObject<QuantScheme>(json);
            Output[] outputs = quantScheme!.Outputs!;

            var nodeCounter = 0;
            foreach (var (var, fakeNode) in _fakeNodesVars)
            {
                nodeCounter++;
                float[] argRange = new float[2];

                // 输入信息的填充
                var parameters = ((Op)((Call)fakeNode).Target).Parameters.ToArray();

                var inputConfigs = new List<QuantConfigData>();
                var outConfigs = new List<QuantConfigData>();
                bool fallbackToCPU = false;
                for (int argIdx = 0; argIdx < ((Call)fakeNode).Arguments.Length; argIdx++)
                {
                    var inputName = (fakeNode.Metadata.OutputNames?[0] ?? $"fakeNode_{nodeCounter}") + $"_input_{argIdx}";
                    var inputInfo = quantScheme.GetInfoByName(inputName);
                    if (inputInfo == null)
                    {
                        fallbackToCPU = true;
                        break;
                    }

                    var dType = DataTypes.FromShortName(inputInfo.DataType!);
                    List<float> inputRange = new();
                    foreach (var range in inputInfo.DataRange!)
                    {
                        inputRange.Add(range.Min);
                        inputRange.Add(range.Max);
                    }

                    inputConfigs.Add(new QuantConfigData(Tensor.From(inputRange.ToArray()), dType));
                }

                if (fallbackToCPU)
                {
                    _fakeNodeConfigs![var] = new QuantConfig(-1);
                    continue;
                }

                // 输出range: 这里的range以及量化方式在后续中不会实际使用到
                int outCounter = 0;
                int maxOutNum = 100;
                for (int outIdx = 0; outIdx < maxOutNum; outIdx++)
                {
                    outCounter++;
                    var outputName = (fakeNode.Metadata.OutputNames?[0] ?? $"fakeNode_{nodeCounter}") + $"_output_{outIdx}";
                    var inputInfo = quantScheme.GetInfoByName(outputName);
                    if (inputInfo == null)
                    {
                        break;
                    }

                    var dType = DataTypes.FromShortName(inputInfo.DataType!);
                    List<float> outRange = new();
                    foreach (var range in inputInfo.DataRange!)
                    {
                        outRange.Add(range.Min);
                        outRange.Add(range.Max);
                    }

                    outConfigs.Add(new QuantConfigData(Tensor.From(outRange.ToArray()), dType));
                }

                var configHeader = new float[] { parameters.Length, outCounter };

                var quantInfo = configHeader.Concat(inputConfigs.SelectMany(x => x.ToRaw())).Concat(outConfigs.SelectMany(x => x.ToRaw())).ToArray();
                var quantConfig = QuantConfig.FromRaw(quantInfo);
                _fakeNodeConfigs![var] = quantConfig;
            }
        }
    }

    private async Task UpdateQuantConfigBySensitivity(List<KeyValuePair<(Var Var, QuantConfig QuantConfig), float>> sensitivity, Tensor groundTruth)
    {
        if (sensitivity.Count == 0)
        {
            return;
        }

        var samples = await _quantizeOptions.CalibrationDataset!.Samples.ToListAsync();
        var sampleFillVarWithConst = new Dictionary<Var, IValue>(samples.First());

        foreach (var (var, config) in _fakeNodeConfigs!)
        {
            if (config.IsEmpty())
            {
                continue;
            }

            sampleFillVarWithConst.Add(var, Value.FromTensor(config.ToRaw()));
        }

        var outDefault = 0;

        // debug:delete
        int debugFakeNode = 0;
        foreach (var (config, _) in sensitivity)
        {
            var curentResults = CompilerServices.Evaluate(((Function)_expr).Body, sampleFillVarWithConst);
            var currentResult = curentResults is TensorValue ? curentResults.AsTensor() : curentResults[outDefault].AsTensor();
            var cosine = Utility.GetCosineSimilarity(MemoryMarshal.Cast<byte, float>(groundTruth.BytesBuffer), MemoryMarshal.Cast<byte, float>(currentResult.BytesBuffer));
            if (cosine > _quantizeOptions.CosineTarget)
            {
                return;
            }

            _fakeNodeConfigs![config.Var] = config.QuantConfig;
            sampleFillVarWithConst[config.Var] = Value.FromTensor(config.QuantConfig.ToRaw());

            Console.ResetColor();
            Console.Write($"try opt: nodes:{sensitivity.Count} current:{++debugFakeNode} targetCosine:");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write($"{_quantizeOptions.CosineTarget}");
            Console.ResetColor();
            Console.Write($" currentCosine:");
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"{cosine}");
            Console.ResetColor();
        }
    }

    private void UpdateQuantTypesByConfig(Dictionary<Expr, Dictionary<ParameterInfo, DataType>> quantTypes, DataType quantType, DataType wQuantType)
    {
        foreach (var (_, fakeNode) in _fakeNodesVars)
        {
            Dictionary<ParameterInfo, DataType> paraTypes = new();
            var parameters = ((Op)((Call)fakeNode).Target).Parameters.ToArray();
            for (int argIdx = 0; argIdx < ((Call)fakeNode).Arguments.Length; argIdx++)
            {
                if (parameters[argIdx].ParameterKind == ParameterKind.Weights)
                {
                    paraTypes.Add(parameters[argIdx], wQuantType);
                }
                else
                {
                    paraTypes.Add(parameters[argIdx], quantType);
                }
            }

            quantTypes.Add(fakeNode, paraTypes);
        }
    }

    private void FillQuantConfig(IDictionary<Expr, ValueRange<float>[]> ranges, Dictionary<Expr, Dictionary<ParameterInfo, DataType>> quantTypes)
    {
        var quantTypeDefault = DataTypes.UInt8;
        foreach (var (var, fakeNode) in _fakeNodesVars)
        {
            float[] argRange = new float[2];

            // 输入信息的填充
            var parameters = ((Op)((Call)fakeNode).Target).Parameters.ToArray();
            var configHeader = new float[] { parameters.Length, ranges[fakeNode].Length };
            var inputConfigs = new List<QuantConfigData>();
            var outConfigs = new List<QuantConfigData>();
            for (int argIdx = 0; argIdx < ((Call)fakeNode).Arguments.Length; argIdx++)
            {
                var quantType = quantTypes[fakeNode].GetValueOrDefault(parameters[argIdx], quantTypeDefault);
                var arg = ((Call)fakeNode).Arguments[argIdx];
                if (arg is not Nncase.IR.None)
                {
                    if (parameters[argIdx].ParameterKind == ParameterKind.Weights)
                    {
                        var dType = quantType;
                        var weights = (TensorConst)((Call)fakeNode).Arguments[argIdx];
                        var weightsValue = weights.Value.ToArray<float>();
                        var oc = weights.CheckedShape[0].FixedValue;
                        var minMaxArr = QuantUtility.GetWeightsRangesByChannel(weightsValue, oc);
                        inputConfigs.Add(new QuantConfigData(Tensor.From(minMaxArr.ToArray(), new[] { oc, 2 }), dType));
                    }
                    else
                    {
                        // 每个parameter的range只有一个，所以这里固定索引为0
                        var dType = quantType;
                        argRange = new float[] { ranges[arg][0].Min, ranges[arg][0].Max };
                        inputConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
                    }
                }
                else
                {
                    var dType = quantTypeDefault;
                    argRange = new float[] { float.MinValue, float.MaxValue };
                    inputConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
                }
            }

            // 输出range: 这里的range以及量化方式在后续中不会实际使用到
            foreach (var outRange in ranges[fakeNode])
            {
                var dType = quantTypeDefault;
                argRange = new float[] { outRange.Min, outRange.Max };
                outConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
            }

            var quantInfo = configHeader.Concat(inputConfigs.SelectMany(x => x.ToRaw())).Concat(outConfigs.SelectMany(x => x.ToRaw())).ToArray();
            var quantConfig = QuantConfig.FromRaw(quantInfo);
            _fakeNodeConfigs![var] = quantConfig;
        }
    }

    private async Task<(List<KeyValuePair<(Var Var, QuantConfig QuantConfig), float>> SensSorted, Tensor GroundTruth)> GetSensitivity(IDictionary<Expr, ValueRange<float>[]> ranges, Dictionary<Expr, Dictionary<ParameterInfo, DataType>> quantTypes)
    {
        UpdateQuantTypesByConfig(quantTypes, DataTypes.UInt8, DataTypes.UInt8);

        var sensitivities = new Dictionary<(Var, QuantConfig), float>();

        var samples = await _quantizeOptions.CalibrationDataset!.Samples.ToListAsync();
        var sampleFillVarWithConst = new Dictionary<Var, IValue>(samples.First());

        foreach (var (var, _) in _fakeNodeConfigs!)
        {
            sampleFillVarWithConst.Add(var, Value.FromConst((float)Math.PI));
        }

        // 多输出的情况下，这里固定取了第一个，后面有更好的策略可以改进
        var outDefault = 0;
        var groundTruths = CompilerServices.Evaluate(((Function)_expr).Body, sampleFillVarWithConst);
        var groundTruth = groundTruths is TensorValue ? groundTruths.AsTensor() : groundTruths[outDefault].AsTensor();

        var quantTypeDefault = DataTypes.UInt8;

        // debug:delete
        int debugFakeNode = 0;
        foreach (var (var, fakeNode) in _fakeNodesVars)
        {
            // debug:delete
            int debugQuantType = 0;
            debugFakeNode++;

            var sampleFillVarWithQuant = new Dictionary<Var, IValue>(sampleFillVarWithConst);
            var quantTypeSupport = ((QuantizeOp)((Call)fakeNode).Target).SupportedQuantType();
            var parameters = ((Op)((Call)fakeNode).Target).Parameters.ToArray();

            var outConfigs = new List<QuantConfigData>();
            foreach (var outRange in ranges[fakeNode])
            {
                var dType = quantTypeDefault;
                var argRange = new float[] { outRange.Min, outRange.Max };
                outConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
            }

            Dictionary<QuantConfig, float> sensitivity = new();
            foreach (var quantType in quantTypeSupport)
            {
                var configHeader = new float[] { parameters.Length, ranges[fakeNode].Length };
                var inputConfigs = new List<QuantConfigData>();
                for (int argIdx = 0; argIdx < ((Call)fakeNode).Arguments.Length; argIdx++)
                {
                    var dType = quantType.GetValueOrDefault(parameters[argIdx], quantTypeDefault);
                    var arg = ((Call)fakeNode).Arguments[argIdx];
                    if (arg is not Nncase.IR.None)
                    {
                        if (parameters[argIdx].ParameterKind == ParameterKind.Weights)
                        {
                            var weights = (TensorConst)((Call)fakeNode).Arguments[argIdx];
                            var weightsValue = weights.Value.ToArray<float>();
                            var oc = weights.CheckedShape[0].FixedValue;
                            var minMaxArr = QuantUtility.GetWeightsRangesByChannel(weightsValue, oc);
                            inputConfigs.Add(new QuantConfigData(Tensor.From(minMaxArr.ToArray(), new[] { oc, 2 }), dType));
                        }
                        else
                        {
                            // 每个parameter的range只有一个，所以这里固定索引为0
                            var argRange = new float[] { ranges[arg][0].Min, ranges[arg][0].Max };
                            inputConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
                        }
                    }
                    else
                    {
                        var argRange = new float[] { float.MinValue, float.MaxValue };
                        inputConfigs.Add(new QuantConfigData(new Tensor<float>(argRange, new[] { 1, 2 }), dType));
                    }
                }

                var quantInfo = configHeader.Concat(inputConfigs.SelectMany(x => x.ToRaw())).Concat(outConfigs.SelectMany(x => x.ToRaw())).ToArray();
                var quantConfig = QuantConfig.FromRaw(quantInfo);

                sampleFillVarWithQuant[var] = Value.FromTensor(quantConfig.ToRaw());
                var currentResults = CompilerServices.Evaluate(((Function)_expr).Body, sampleFillVarWithQuant);
                var currentResult = currentResults is TensorValue ? currentResults.AsTensor() : currentResults[outDefault].AsTensor();
                var cosine = Utility.GetCosineSimilarity(MemoryMarshal.Cast<byte, float>(groundTruth.BytesBuffer), MemoryMarshal.Cast<byte, float>(currentResult.BytesBuffer));

                sensitivities[(var, quantConfig)] = cosine;

                // debug:delete
                // sensitivities[(var, quantConfig)] = (float)((float)(cosine + (0.0001 * debugFakeNode) + (debugQuantType * 0.00001)) - 0.5);
                DateTime now = DateTime.Now;
                string timestamp = now.ToString("HH:mm");
                Console.WriteLine($"sensitivity: nodes:{debugFakeNode}/{_fakeNodesVars.Count} types:{++debugQuantType}/{quantTypeSupport.Count} time:{timestamp} cosine:{cosine}");
            }
        }

        // var sensSorted = sensitivities
        // .OrderBy(entry => entry.Value.Values.Min())
        // .ToDictionary(
        //     entry => entry.Key,
        //     entry => entry.Value.OrderBy(kv => kv.Value).ToDictionary(kv => kv.Key, kv => kv.Value));
        var sensSorted = sensitivities.OrderBy(pair => pair.Value).ToList();
        AddSensitivityForFallback(sensSorted);

        return (sensSorted, groundTruth);
    }

    private void AddSensitivityForFallback(List<KeyValuePair<(Var Var, QuantConfig QuantConfig), float>> sensSorted)
    {
        // 由于某些节点的量化损失很大，因此我们尽早把这些节点回退到cpu，否则会导致搜索速度很慢。这里创建了一个列表记录了需要在哪些位置插入回退到CPU的节点。这里之所以先记录节点，然后再倒叙插入节点，是因为插入节点的时候列表长度改变，导致InvalidOperationException
        var toInsert = new List<(int, KeyValuePair<(Var, QuantConfig), float>)>();

        // 遍历列表，记下需要插入新元素的位置
        for (int i = 0; i < sensSorted.Count; i++)
        {
            if (sensSorted[i].Value < 0.95f)
            {
                // 构造新的KeyValuePair
                var newPair = new KeyValuePair<(Var, QuantConfig), float>((sensSorted[i].Key.Var, new QuantConfig(-1)), 2.0f);

                // 记录位置和新元素
                toInsert.Add((i + 1, newPair));
            }
        }

        // 反向遍历toInsert列表，这样我们就不会因为插入操作改变索引而出错
        for (int i = toInsert.Count - 1; i >= 0; i--)
        {
            var insertAt = toInsert[i].Item1;
            var elementToInsert = toInsert[i].Item2;
            sensSorted.Insert(insertAt, elementToInsert);
        }

        // 以下操作是最后的补刀：给所有节点回退到cpu的机会。
        var ticketsforVar = new Dictionary<Var, int>();
        int idx = 0;
        foreach (var (var, _) in sensSorted)
        {
            idx++;
            if (ticketsforVar.TryGetValue(var.Var, out var tickets))
            {
                ticketsforVar[var.Var] = idx + tickets;
            }
            else
            {
                ticketsforVar[var.Var] = idx;
            }
        }

        var ticketsSorted = ticketsforVar.OrderBy(x => x.Value).ToList();

        foreach (var (var, _) in ticketsSorted)
        {
            sensSorted.Add(new KeyValuePair<(Var, QuantConfig), float>((var, new QuantConfig(-1)), 2.0f));
        }
    }

    private void ExportQuantScheme()
    {
        var quantScheme = new QuantScheme();
        quantScheme.Version = "1.0";
        List<Output> outputs = new();
        int nodeCounter = 0;
        foreach (var (var, fakeNode) in _fakeNodesVars)
        {
            nodeCounter++;
            var quantConfig = _fakeNodeConfigs![var];
            var parameters = ((Op)((Call)fakeNode).Target).Parameters.ToArray();
            for (int iPara = 0; iPara < quantConfig.GetInputNum(); iPara++)
            {
                var inputRange = quantConfig.GetInputRange(parameters[iPara]).ToArray();
                var dataRange = Enumerable.Range(0, inputRange.Length / 2)
                .Select(j => new ValueRange<float>
                {
                    Min = inputRange[j * 2],
                    Max = inputRange[(j * 2) + 1],
                }).ToList();

                var output = new Output
                {
                    // 这里之所以再加一个ID，是因为同一个OP有可能会作为多个OP的输入，并且量化方式不一定一致，因此这里加ID以便区分。
                    Name = (fakeNode.Metadata.OutputNames?[0] ?? $"fakeNode_{nodeCounter}") + $"_input_{iPara}",
                    DataType = quantConfig.GetInputQuantType(parameters[iPara]).ToString(),
                    DataRangeMode = parameters[iPara].ParameterKind == ParameterKind.Weights ? "by_channel" : "by_tensor",
                    DataRange = dataRange.ToArray(),
                };

                outputs.Add(output);
            }

            for (int iPara = 0; iPara < quantConfig.GetOutputNum(); iPara++)
            {
                var outputRange = quantConfig.GetOutputRange(iPara).ToArray();
                var dataRange = Enumerable.Range(0, outputRange.Length / 2)
                .Select(j => new ValueRange<float>
                {
                    Min = outputRange[j * 2],
                    Max = outputRange[(j * 2) + 1],
                }).ToList();

                var output = new Output
                {
                    Name = (fakeNode.Metadata.OutputNames?[iPara] ?? $"fakeNode_{nodeCounter}") + $"_output_{iPara}",
                    DataType = quantConfig.GetOutputQuantType(iPara).ToString(),
                    DataRangeMode = "by_tensor",
                    DataRange = dataRange.ToArray(),
                };

                outputs.Add(output);
            }
        }

        quantScheme.Outputs = outputs.ToArray();

        var quantSchemeString = JsonConvert.SerializeObject(quantScheme, Newtonsoft.Json.Formatting.Indented);
        _quantizeOptions.QuantSchemeInnerCheck = quantSchemeString;
        if (Path.Exists(DumpScope.Current.Directory))
        {
            File.WriteAllText(Path.Join(DumpScope.Current.Directory, "..", "..", "QuantScheme.json"), quantSchemeString);
        }
    }

    private Expr ExtractExpr()
    {
        Func<ENode, bool> fakeCallFilter = node =>
        {
            return node.Expr is Call call &&
                            (call.Target is QuantizeOp);
        };

        Func<ENode, Tuple<ENode, ENode>> replaceUninitlizedWithVar = node =>
        {
            var parameters = ((Call)node.Expr).Arguments.ToArray();
            parameters[^1] = new Var("placeholder", new TensorType(DataTypes.Float32, Shape.Unknown(1)));
            var newCall = ((Call)node.Expr).With(arguments: parameters, metadata: ((Call)node.Expr).Metadata);
            var newEnode = ENode.Create(newCall, Array.Empty<EClass>());
            _fakeNodeConfigs?.Add((Var)parameters[^1], default!); // config detail info will be update later.
            return new Tuple<ENode, ENode>(node, newEnode);
        };

        var callPairs = _graph.Nodes.Where(node => fakeCallFilter(node)).Select(node => replaceUninitlizedWithVar(node)).ToArray();

        foreach ((var nodeWithUninitlized, var nodeWithVar) in callPairs)
        {
            var rangeEclass = _graph.Add(nodeWithVar.Expr);
            var rangeOfEclass = _graph.Find(nodeWithUninitlized);
            _graph.Union(rangeOfEclass, rangeEclass);
        }

        _graph.Rebuild();

        var expr = _graph.Extract(_graph.Root!, null, Array.Empty<EGraphExtractConstrains>());

        ExprCollector.Collect(expr).OfType<Call>().Where(call => call.Target is QuantizeOp).ToList().ForEach(fakeCall => _fakeNodesVars![(Var)fakeCall.Arguments[^1]] = fakeCall);

        var keys1 = _fakeNodesVars.Keys;
        var keys2 = _fakeNodeConfigs!.Keys;
        Trace.Assert(keys1.All(key => keys2.Contains(key)));

        return expr;
    }

    private void AssignRanges()
    {
        if (_fakeNodeConfigs is not null)
        {
            foreach (var (var, config) in _fakeNodeConfigs)
            {
                if (!config.IsEmpty())
                {
                    var eNode = ENode.Create(var, Array.Empty<EClass>());
                    var rangeEclass = _graph.Add(config.ToRaw());
                    var rangeOfEclass = _graph.Find(eNode);
                    _graph.Union(rangeOfEclass, rangeEclass);
                }
                else
                {
                    var eNode = ENode.Create(var, Array.Empty<EClass>());
                    var rangeEclass = _graph.Add(new QuantConfig(-1).ToRaw());
                    var rangeOfEclass = _graph.Find(eNode);
                    _graph.Union(rangeOfEclass, rangeEclass);
                }
            }

            _graph.Rebuild();

            // var debugExpr = _graph.Extract(_graph.Root!, null, out _);
        }

        var dumpPath = Path.Join(DumpScope.Current.Directory, "..", "..", "..", "/");
        EGraphPrinter.DumpEgraphAsDot(_graph, dumpPath + "_graph_after.dot");
    }

    private async Task<IDictionary<Expr, ValueRange<float>[]>> GetRangesAsync(ICalibrationDatasetProvider calibrationDataset)
    {
        var ranges = new Dictionary<Expr, ValueRange<float>[]>(ReferenceEqualityComparer.Instance);
        await RunPassAsync(calibrationDataset, (values) =>
        {
            foreach (var value in values)
            {
                int globalVarIndex = 0;
                if (value.Key is Call && ((Call)value.Key).Arguments[^1] is Var)
                {
                    globalVarIndex = ((Var)((Call)value.Key).Arguments[^1]).GlobalVarIndex;
                }

                try
                {
                    var outputNum = value.Value.AsTensors().Length;
                    var outRange = new ValueRange<float>[outputNum];
                    for (int i = 0; i < outputNum; i++)
                    {
                        var tensor = Tensor.From(value.Value.AsTensors()[i].ToArray<float>());
                        var range = GetMinMax(tensor);
                        if (ranges.TryGetValue(value.Key, out var oldRange))
                        {
                            ranges[value.Key][i] = oldRange[i].Union(range);
                        }
                        else
                        {
                            outRange[i] = range;
                            ranges.Add(value.Key, outRange);
                        }
                    }
                }
                catch (Exception)
                {
                    continue;
                }
            }
        });
        return ranges;
    }

    private Dictionary<Expr, IValue> GetMemo(Dictionary<Var, IValue> sample)
    {
        if (_fakeNodeConfigs is not null && _fakeNodeConfigs.Count != 0)
        {
            foreach (var (var, _) in _fakeNodeConfigs!)
            {
                sample.Add(var, Value.FromConst((float)Math.PI));
            }
        }

        return EvaluatorUtil.GetMemo(_expr, sample);
    }

    private async Task RunPassAsync(ICalibrationDatasetProvider calibrationDataset, Action<IReadOnlyDictionary<Expr, IValue>> func)
    {
        await foreach (Dictionary<Var, IValue> sample in calibrationDataset.Samples)
        {
            var memo = GetMemo(sample);
            func(memo);
            GC.Collect();
        }
    }

    private async Task<IDictionary<Expr, QuantizeHistogram<float>[]>> GetHistogramsAsync(ICalibrationDatasetProvider calibrationDataset, IDictionary<Expr, ValueRange<float>[]> ranges, int srcBinSize, int dstBinSize)
    {
        var histograms = new Dictionary<Expr, QuantizeHistogram<float>[]>(ReferenceEqualityComparer.Instance);
        histograms = ranges.ToDictionary(range => range.Key, range => Enumerable.Range(0, range.Value.Length)
                       .Select(i => new QuantizeHistogram<float>(
                           new List<float>(new float[srcBinSize]),
                           new List<float>(new float[srcBinSize]))).ToArray());

        await RunForHistogramsAsync(calibrationDataset, ranges, childrenValues =>
        {
            foreach (var (key, value) in childrenValues)
            {
                for (int i = 0; i < value.Length; i++)
                {
                    var r = ranges[key][i].Max - ranges[key][i].Min;
                    var srcBinInterval = r / srcBinSize;

                    var childrenTensor = value[i].Cast<float>();
                    var childrenBuffer = childrenTensor.Buffer.Span;
                    var valueRange = ranges[key][i];
                    var histogram = histograms[key][i];
                    foreach (var buf in childrenBuffer)
                    {
                        var r_index = (buf - valueRange.Min) / srcBinInterval;
                        var index = (int)Math.Clamp((float)r_index, 0F, (float)srcBinSize - 1);
                        histogram.SrcBin[index]++;
                    }
                }
            }
        });
        return histograms;
    }

    private async Task RunForHistogramsAsync(ICalibrationDatasetProvider calibrationDataset, IDictionary<Expr, ValueRange<float>[]> ranges, Action<IReadOnlyDictionary<Expr, Tensor[]>> func)
    {
        await foreach (Dictionary<Var, IValue> sample in calibrationDataset.Samples)
        {
            var memoTensors = new Dictionary<Expr, Tensor[]>();
            var memo = GetMemo(sample);
            foreach (var (key, _) in ranges)
            {
                try
                {
                    memoTensors[key] = memo[key].AsTensors();
                }
                catch (Exception)
                {
                    continue;
                }
            }

            func(memoTensors);
            GC.Collect();
        }
    }

    private ValueRange<float> FixUpRange(ValueRange<float> range, bool symmetric = false)
    {
        if (symmetric)
        {
            var r = Math.Max(0.01f, Math.Max(Math.Abs(range.Min), Math.Abs(range.Max)));
            return new ValueRange<float>(-r, r);
        }
        else
        {
            if (range.Max < 0)
            {
                range.Max = 0;
            }

            if (range.Min > 0)
            {
                range.Min = 0;
            }

            var r = range.Max - range.Min;
            if (r < 1e-6f)
            {
                r = 0.1f;
            }

            range.Max = range.Min + r;
        }

        return range;
    }

    private IDictionary<Expr, ValueRange<float>[]> GetOptRanges(IDictionary<Expr, QuantizeHistogram<float>[]> histograms, IDictionary<Expr, ValueRange<float>[]> ranges, int srcBinSize, int dstBinSize, CalibMethod calibrationMethod)
    {
        var optRanges = new Dictionary<Expr, ValueRange<float>[]>(ReferenceEqualityComparer.Instance);
        if (calibrationMethod == CalibMethod.Kld)
        {
            foreach (KeyValuePair<Expr, QuantizeHistogram<float>[]> histogram in histograms)
            {
                var optRange = new ValueRange<float>[histogram.Value.Length];
                for (int i = 0; i < histogram.Value.Length; i++)
                {
                    histogram.Value[i].SrcBin = Smooth(histogram.Value[i].SrcBin);
                    var minKld = float.MaxValue;
                    var r = ranges[histogram.Key][i].Max - ranges[histogram.Key][i].Min;
                    var srcBinInterval = r / srcBinSize;
                    var betterThreshold = new Tuple<int, int>(0, srcBinSize);
                    var zeroThreshold = (int)Math.Clamp((0 - ranges[histogram.Key][i].Min) / srcBinInterval, 0, srcBinSize - 1);

                    // range max first
                    int lowerThreshold = 0;
                    var srcBin = histogram.Value[i].SrcBin;
                    for (int upperThreshold = srcBinSize; upperThreshold >= dstBinSize && upperThreshold >= zeroThreshold; upperThreshold -= dstBinSize)
                    {
                        GetKldOptRanges(lowerThreshold, upperThreshold, dstBinSize, ref srcBin, ref minKld, ref betterThreshold);
                    }

                    // range min
                    minKld = float.MaxValue;
                    int upperThreshold2 = betterThreshold.Item2;
                    for (int lowerThreshold2 = 0; lowerThreshold2 <= zeroThreshold && lowerThreshold2 <= upperThreshold2 - dstBinSize; lowerThreshold2 += dstBinSize)
                    {
                        GetKldOptRanges(lowerThreshold2, upperThreshold2, dstBinSize, ref srcBin, ref minKld, ref betterThreshold);
                    }

                    var optMin = ((betterThreshold.Item1 - 0.5f) * srcBinInterval) + ranges[histogram.Key][i].Min;
                    var optMax = ((betterThreshold.Item2 + 0.5f) * srcBinInterval) + ranges[histogram.Key][i].Min;
                    optRange[i] = new ValueRange<float>(optMin, optMax);
                }

                optRanges.Add(histogram.Key, optRange);
            }

            return optRanges;
        }
        else
        {
            throw new ArgumentException("Invalid calibration method.");
        }
    }
}
