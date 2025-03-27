// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Google.Protobuf.Collections;
using LanguageExt;
using NetFabric.Hyperlinq;
using Newtonsoft.Json.Linq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer;

public class ModelInitContext
{
    public Dictionary<string, object>? Config = new Dictionary<string, object>();

    public Dictionary<string, Tensor>? ConstTensors = new Dictionary<string, Tensor>();

    public Dictionary<string, Expr> Outputs = new Dictionary<string, Expr>();

    // private readonly Dictionary<Var, Expr[]>? _varMap;
    public List<Var?>? Inputs = new List<Var?>();

    public Dictionary<string, Var>? DynVarMap = new Dictionary<string, Var>();

    public Dictionary<string, int>? FixVarMap = new Dictionary<string, int>();

    public CompileSession? CompileSession { get; set; }
}

public partial class HuggingFaceImporter : BaseImporter
{
    private readonly HuggingFaceModel _model;
    private ModelInitContext _modelContext = new();

    public HuggingFaceImporter(string huggingFaceDir, CompileSession compileSession)
        : base(compileSession)
    {
        var config = HuggingFaceUtils.GetConfigInfo(Path.Combine(huggingFaceDir, "config.json"));
        var tmp_config = HuggingFaceUtils.GetConfigInfo(Path.Combine(huggingFaceDir, "generation_config.json"));
        foreach (var pair in tmp_config)
        {
            config[pair.Key] = pair.Value;
        }

        var constTensors = new Dictionary<string, Tensor>();
        if (File.Exists(Path.Combine(huggingFaceDir, "model.safetensors.index.json")))
        {
            string[] files = Directory.GetFiles(huggingFaceDir, "*.safetensors");
            foreach (string file in files)
            {
                string fileName = Path.GetFileName(file);
                if (IsModelFile(fileName))
                {
                    Console.WriteLine($"find safetensor file: {fileName}");
                    var tmpConst = HuggingFaceUtils.GetAllWeights(Path.Combine(huggingFaceDir, fileName));
                    foreach (var item in tmpConst)
                    {
                        constTensors[item.Key] = item.Value;
                    }
                }
            }
        }
        else
        {
            constTensors = HuggingFaceUtils.GetAllWeights(Path.Combine(huggingFaceDir, "model.safetensors"));
        }

        _modelContext!.Config = config;
        _modelContext!.ConstTensors = constTensors;
        _modelContext!.CompileSession = compileSession;
        switch (config.GetNestedValue<string>("architectures", 0))
        {
            case "Qwen2ForCausalLM":
                _model = new Qwen2();
                break;
            case "LlamaForCausalLM":
                _model = new Llama3_2();
                break;
            default:
                throw new NotImplementedException();
        }

        _model!.Initialize(_modelContext, huggingFaceDir);
    }

    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateInputs()
    {
        return _model.CreateInputs();
    }

    protected override void ConvertOp()
    {
        _model.VisitForCausalLM();
    }

    protected override Expr CreateOutputs()
    {
        return _model.CreateOutputs();
    }

    private static bool IsModelFile(string fileName)
    {
        if (!fileName.StartsWith("model-") || !fileName.EndsWith(".safetensors"))
        {
            return false;
        }

        string middlePart = fileName.Substring("model-".Length, fileName.Length - "model-".Length - ".safetensors".Length);

        string[] parts = middlePart.Split('-');
        if (parts.Length != 3 || parts[1] != "of")
        {
            return false;
        }

        if (!int.TryParse(parts[0], out _) || !int.TryParse(parts[2], out _))
        {
            return false;
        }

        return true;
    }
}
