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

    private List<Var>? _inputs;
    private readonly Dictionary<string, Expr>? _outputs = new Dictionary<string, Expr> { };
    private Dictionary<string, Var> _dynVarMap;
    private Dictionary<string, int> _fixVarMap;

    private readonly Dictionary<Var, Expr[]> _varMap;

    public HuggingFaceImporter(string huggingFaceDir, CompileSession compileSession)
        : base(compileSession)
    {
        // 读取 config.json 文件
        _config = HuggingFaceUtils.GetConfigInfo(Path.Combine(huggingFaceDir, "config.json"));
        _constTensors = HuggingFaceUtils.GetAllWeights(Path.Combine(huggingFaceDir, "model.safetensors"));
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
        switch (_config!["architectures"]!)
        {
            case "Qwen2ForCausalLM":
                return Qwen2CreateOutputs();

            default:
                throw new NotImplementedException();
        }
    }
}
