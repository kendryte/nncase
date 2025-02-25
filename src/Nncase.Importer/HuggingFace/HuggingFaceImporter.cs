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
    private Dictionary<string, object>? _config;
    private Dictionary<string, Tensor>? _constTensors;

    private List<Var>? _inputs;
    private List<Var>? _outputs;
    private Dictionary<string, Var> _dynVarMap = new();
    private Dictionary<string, int> _fixVarMap = new();

    public HuggingFaceImporter(string huggingFaceDir, CompileSession compileSession)
        : base(compileSession)
    {
        // 读取 config.json 文件
        _config = HuggingFaceUtils.getConfigInfo(Path.Combine(huggingFaceDir, "config.json"));
        _constTensors = HuggingFaceUtils.getAllWeights(Path.Combine(huggingFaceDir, "model.safetensors"));
    }

    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateInputs()
    {
        throw new NotImplementedException();
        switch (_config!["architectures"]!)
        {
            case "Qwen2ForCausalLM":
                Qwen2CreateInputs();
                break;
            default:
                throw new NotImplementedException();
        }
    }

    protected override void ConvertOp()
    {
        switch (_config!["architectures"]!)
        {
            case "Qwen2ForCausalLM":
                _config["pad_token_id"] = 0;
                Debug.Assert(_constTensors != null, nameof(_constTensors) + " != null");
                VisitQwen2ForCausalLM();
                break;
            default:
                throw new NotImplementedException();
        }
    }

    protected override Expr CreateOutputs()
    {
        throw new NotImplementedException();
    }
}
