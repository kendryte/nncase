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

public sealed partial class HuggingFaceImporter : BaseImporter
{
    private readonly Dictionary<string, object>? _config;
    private readonly Dictionary<string, Tensor>? _constTensors;
    private readonly Dictionary<string, Expr>? _outputs = new Dictionary<string, Expr> { };

    // private readonly Dictionary<Var, Expr[]>? _varMap;
    private List<Var?>? _inputs;
    private Dictionary<string, Var>? _dynVarMap;
    private Dictionary<string, int>? _fixVarMap;

    public HuggingFaceImporter(string huggingFaceDir, CompileSession compileSession)
        : base(compileSession)
    {
        // TODO: restructure for reading saftensors
        // 读取 config.json 文件
        _config = HuggingFaceUtils.GetConfigInfo(Path.Combine(huggingFaceDir, "config.json"));
        var tmp_config = HuggingFaceUtils.GetConfigInfo(Path.Combine(huggingFaceDir, "generation_config.json"));
        foreach (var pair in tmp_config)
        {
            _config[pair.Key] = pair.Value;
        }

        if (File.Exists(Path.Combine(huggingFaceDir, "model.safetensors.index.json")))
        {
            _constTensors = new Dictionary<string, Tensor>();
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
                        _constTensors[item.Key] = item.Value;
                    }
                }
            }
        }
        else
        {
            _constTensors = HuggingFaceUtils.GetAllWeights(Path.Combine(huggingFaceDir, "model.safetensors"));
        }
    }

    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateInputs()
    {
        // throw new NotImplementedException();
        switch (_config!["architectures"]!)
        {
            case "Qwen2ForCausalLM":
                return Qwen2CreateInputs();

            default:
                throw new NotImplementedException();
        }
    }

    protected override void ConvertOp()
    {
        switch (_config!["architectures"]!)
        {
            case "Qwen2ForCausalLM":
                Debug.Assert(_constTensors != null, nameof(_constTensors) + " != null");
                VisitQwen2ForCausalLM();
                break;
            default:
                throw new NotImplementedException();
        }
    }

    protected override Expr CreateOutputs()
    {
        switch (_config!["architectures"]!)
        {
            case "Qwen2ForCausalLM":
                return Qwen2CreateOutputs();

            default:
                throw new NotImplementedException();
        }
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
