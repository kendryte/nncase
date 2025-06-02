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

        var constTensors = new Dictionary<string, Tensor>();
        if (File.Exists(Path.Combine(huggingFaceDir, "model.safetensors.index.json")))
        {
            string[] files = Directory.GetFiles(huggingFaceDir, "*.safetensors");
            foreach (string file in files)
            {
                string fileName = Path.GetFileName(file);
                if (IsModelFile(fileName))
                {
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
        _modelContext!.ImportOptions = importOptions;
        _modelContext!.CompileSession = compileSession;
        switch (config.GetNestedValue<string>("architectures", 0))
        {
            case "Qwen2ForCausalLM":
                _model = new Qwen2();
                break;
            case "Qwen3ForCausalLM":
                _model = new Qwen3();
                break;
            case "LlamaForCausalLM":
                _model = new Llama3_2();
                break;
            case "GlmForCausalLM":
                _model = new Glm4V9B();
                break;
            default:
                throw new NotImplementedException();
        }

        _model!.Initialize(_modelContext, huggingFaceDir);
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
