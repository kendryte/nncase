// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Google.OrTools.ConstraintSolver;
using Google.Protobuf.WellKnownTypes;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.Quantization;
using Nncase.Studio.Util;

namespace Nncase.Studio.ViewModels;

public enum PreprocessMode
{
    Custom,
}

public partial class CompileOptionViewModel : ViewModelBase
{
    [Required]
    [ObservableProperty]
    private string _inputFile = string.Empty;

    [ObservableProperty]
    private string _dumpDir = string.Empty;

    [ObservableProperty]
    private bool _preprocess;

    [ObservableProperty]
    private bool _quantize = true;

    [ObservableProperty]
    private bool _shapeBucket;

    [ObservableProperty]
    private bool _mixQuantize = false;

    [ObservableProperty]
    private string _inputFormat = string.Empty;

    [ObservableProperty]
    private string _target;

    [ObservableProperty]
    private PreprocessMode _preprocessMode = PreprocessMode.Custom;

    public CompileOptionViewModel(ViewModelContext context)
    {
        // skip None
        DumpFlagsList = new ObservableCollection<DumpFlags>(System.Enum.GetValues<DumpFlags>().Skip(1).ToList());
        TargetList = new ObservableCollection<string>(new[] { "cpu", "k230" });

        _target = TargetList[0];
        Context = context;
        var list = new[] { PreprocessMode.Custom };
        PreprocessModeList = new(list);
    }

    public ObservableCollection<DumpFlags> DumpFlagSelected { get; set; } = new();

    public ObservableCollection<DumpFlags> DumpFlagsList { get; set; }

    public ObservableCollection<string> TargetList { get; set; }

    public ObservableCollection<PreprocessMode> PreprocessModeList { get; set; }

    [RelayCommand]
    public async Task SetDumpDir()
    {
        var folder = await Context.OpenFolder(PickerOptions.FolderPickerOpenOptions);
        if (folder != string.Empty)
        {
            DumpDir = folder;
        }
    }

    public override void UpdateViewModelCore(CompileConfig config)
    {
        InputFile = config.CompileOption.InputFile;
        InputFormat = config.CompileOption.InputFormat;
        DumpDir = config.CompileOption.DumpDir;
        Target = config.Target;
        Preprocess = config.CompileOption.PreProcess;
        MixQuantize = config.MixQuantize;
        ShapeBucket = config.EnableShapeBucket;
        Quantize = config.UseQuantize;

        PreprocessMode = config.PreprocessMode;
        DumpFlagSelected = new(config.DumpFlags.ToArray());
    }

    public override void UpdateConfig(CompileConfig config)
    {
        config.CompileOption.DumpDir = DumpDir;
        config.DumpFlags = DumpFlagSelected.ToArray();
        config.CompileOption.DumpFlags = DumpFlagSelected.Aggregate(DumpFlags.None, (flag, sum) => flag | sum);
        config.Target = Target;
        config.CompileOption.PreProcess = Preprocess;
        config.MixQuantize = MixQuantize;
        config.EnableShapeBucket = ShapeBucket;
        config.UseQuantize = Quantize;
        config.PreprocessMode = PreprocessMode;
        if (Quantize == false)
        {
            config.CompileOption.QuantizeOptions.ModelQuantMode = ModelQuantMode.NoQuant;
        }
    }

    public override List<string> CheckViewModel()
    {
        var list = new List<string>();
        if (DumpDir == string.Empty)
        {
            list.Add("DumpDir can't be empty");
        }

        return list;
    }
}
