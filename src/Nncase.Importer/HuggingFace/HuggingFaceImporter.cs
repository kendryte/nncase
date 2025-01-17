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
using TorchSharp;
using static TorchSharp.torch.nn;

namespace Nncase.Importer;

public sealed partial class HuggingFaceImporter : BaseImporter
{
    private string _modelDir;
    private Dictionary<string, object> _config;
	private List<string> _modelArchitectures;

    private Dictionary<string, Nncase.Tensor>? _constTensors;

    private Dictionary<string, Var> _dynVarMap = new();
    private Dictionary<string, int> _fixVarMap = new();

    public HuggingFaceImporter(string huggingFaceDir, CompileSession compileSession)
        : base(compileSession)
    {
        _modelDir = huggingFaceDir;

        // 读取 config.json 文件
        getConfigInfo(Path.Combine(huggingFaceDir, "config.json"));
        getAllWeights(Path.Combine(huggingFaceDir, "model.safetensors"));


        if (String.Equals(_config["architectures"], "Qwen2ForCausalLM"))
        {
            _modelArchitectures = new List<string>() {"Qwen2Model", "Linear"};
			//{ "Embedding", "Qwen2DecoderLayer", "Qwen2MLP", "Qwen2RMSNorm", "Qwen2RotaryEmbedding" };
        }
    }

    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateInputs()
    {
        throw new NotImplementedException();
    }

    protected override void ConvertOp()
    {
        foreach (var architecture in _modelArchitectures)
        {
			Visit(architecture);
        }
    }

    protected override Expr CreateOutputs()
    {
        throw new NotImplementedException();
    }

	private void Visit(string op)
    {
        switch (op)
        {
            case "Qwen2Model":
                VisitQwen2Model(_config, _constTensors);
                break;
        }
    }

    private void getConfigInfo(string path)
    {
        if (File.Exists(path))
        {
            var configJson = File.ReadAllText(path);
            _config = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, object>>(configJson);
            foreach (var key in _config.Keys.ToList())
            {
                if (_config[key] is JArray jArray)
                {
                    _config[key] = string.Join(", ", jArray.Select(token => token.ToString()));
                }
            }
        }
        else
        {
            throw new FileNotFoundException($"{_config?["architectures"]}'s config.json not found in the specified directory.", path);
        }
    }

    private void getAllWeights(string path)
    {
        var constTensor = HuggingFaceUtils.LoadStateDict(path);
        foreach (var item in constTensor)
        {
            Console.WriteLine($"{item.Key}");
            if (item.Value is Tensor tensor)
            {
                _constTensors ??= new();
                _constTensors.Add(item.Key, tensor.CastTo(DataTypes.Float32));
            }
        }
    }


}
