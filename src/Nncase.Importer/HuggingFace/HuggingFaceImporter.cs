// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using Nncase.IR;

namespace Nncase.Importer;

public class ModelInitContext
{
    private Dictionary<string, object>? _config = new Dictionary<string, object>();

    private Dictionary<string, Tensor>? _constTensors = new Dictionary<string, Tensor>();

    private Dictionary<string, Expr> _outputs = new Dictionary<string, Expr>();

    private List<Var?>? _inputs = new List<Var?>();

    private Dictionary<string, DimVar>? _dynVarMap = new Dictionary<string, DimVar>();

    private Dictionary<string, int>? _fixVarMap = new Dictionary<string, int>();

    private ImportOptions? _importOptions;

    private CompileSession? _compileSession;

    // 公共属性
    public Dictionary<string, object>? Config
    {
        get { return _config; }
        set { _config = value; }
    }

    public Dictionary<string, Tensor>? ConstTensors
    {
        get { return _constTensors; }
        set { _constTensors = value; }
    }

    public Dictionary<string, Expr> Outputs
    {
        get { return _outputs; }
        set { _outputs = value; }
    }

    public List<Var?>? Inputs
    {
        get { return _inputs; }
        set { _inputs = value; }
    }

    public Dictionary<string, DimVar>? DynVarMap
    {
        get { return _dynVarMap; }
        set { _dynVarMap = value; }
    }

    public Dictionary<string, int>? FixVarMap
    {
        get { return _fixVarMap; }
        set { _fixVarMap = value; }
    }

    public ImportOptions? ImportOptions
    {
        get { return _importOptions; }
        set { _importOptions = value; }
    }

    public CompileSession? CompileSession
    {
        get { return _compileSession; }
        set { _compileSession = value; }
    }
}

public partial class HuggingFaceImporter : BaseImporter
{
    private readonly HuggingFaceModel _model;
    private readonly ModelInitContext _modelContext = new();

    public HuggingFaceImporter(string huggingFaceDir, ImportOptions importOptions, CompileSession compileSession)
        : base(compileSession)
    {
        var config = HuggingFaceUtils.GetConfigInfo(Path.Combine(huggingFaceDir, "config.json"));
        var tmp_config = HuggingFaceUtils.GetConfigInfo(Path.Combine(huggingFaceDir, "generation_config.json"));
        foreach (var pair in tmp_config)
        {
            config[pair.Key] = pair.Value;
        }

        if (importOptions.HuggingFaceOptions.NumLayers != -1)
        {
            if (importOptions.HuggingFaceOptions.NumLayers < (long)config["num_hidden_layers"])
            {
                Console.WriteLine($"HuggingFaceOptions.NumLayers is set to [{importOptions.HuggingFaceOptions.NumLayers}], which is less than num_hidden_layers [{(long)config["num_hidden_layers"]}] in the huggingface model config.");
                config["num_hidden_layers"] = (long)importOptions.HuggingFaceOptions.NumLayers;
            }
            else
            {
                throw new ArgumentException($"HuggingFaceOptions.NumLayers [{importOptions.HuggingFaceOptions.NumLayers}] must be set to a value less than or equal to num_hidden_layers [{(long)config["num_hidden_layers"]}] in the huggingface model config.");
            }
        }

        _modelContext.Config = config;
        _modelContext.ImportOptions = importOptions;
        _modelContext.CompileSession = compileSession;

        var architectures = config.GetNestedValue<string>("architectures", 0);
        _model = architectures switch
        {
            "Qwen2ForCausalLM" => new Qwen2(),
            "Qwen3ForCausalLM" => new Qwen3(),
            "LlamaForCausalLM" => new Llama3_2(),
            "GlmForCausalLM" => new Glm4V9B(),
            _ => throw new NotImplementedException($"Architecture {architectures} is not supported"),
        };

        _model.Initialize(_modelContext, huggingFaceDir);
    }

    protected override (IEnumerable<IVar> Inputs, Dictionary<IVar, Dimension[]> VarMap) CreateInputs()
    {
        return _model.CreateInputs();
    }

    protected override void ConvertOp()
    {
        _model.VisitForCausalLM();
    }

    protected override BaseExpr CreateOutputs()
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
