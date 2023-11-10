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

namespace Nncase.Studio.ViewModels;

public partial class CompileOptionViewModel : ViewModelBase
{
    [Required]
    [ObservableProperty]
    private string _inputFile = string.Empty;

    [ObservableProperty]
    private string _dumpDir;

    // 带有参数的正则验证？？？
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

    public CompileOptionViewModel(ViewModelContext context)
    {
        // skip None
        DumpFlagsList = new ObservableCollection<DumpFlags>(System.Enum.GetValues<DumpFlags>().Skip(1).ToList());
        TargetList = new ObservableCollection<string>(new[] { "cpu", "k230" });

        // todo: k230 should set dll path
        _target = TargetList[0];
        DumpDir = Path.Join(Directory.GetCurrentDirectory(), "nncase_dump");
        this.Context = context;
    }

    public ObservableCollection<DumpFlags> DumpFlagSelected { get; set; } = new();

    public ObservableCollection<DumpFlags> DumpFlagsList { get; set; }

    public ObservableCollection<string> TargetList { get; set; }

    [RelayCommand]
    public async Task SetDumpDir()
    {
        var folder = await Context.OpenFolder(PickerOptions.FolderPickerOpenOptions);
        if (folder != string.Empty)
        {
            DumpDir = folder;
        }
    }

    [RelayCommand]
    public void ShowPreprocess()
    {
        var pre = Context.ViewModelLookup(typeof(PreprocessViewModel))!;
        if (Preprocess)
        {
            Context.InsertPage(pre, this, 1);
        }
        else
        {
            Context.RemovePage(pre);
        }
    }

    [RelayCommand]
    public void ShowQuantize()
    {
        // todo: refactor to ShowViewModel
        var quant = Context.ViewModelLookup(typeof(QuantizeViewModel))!;
        var compile = Context.ViewModelLookup(typeof(CompileViewModel))!;
        if (Quantize)
        {
            Context.InsertPage(quant, compile);
        }
        else
        {
            Context.RemovePage(quant);
        }
    }

    [RelayCommand]
    public void ShowShapeBucket()
    {
        var shapeBucket = Context.ViewModelLookup(typeof(ShapeBucketViewModel))!;
        var compile = Context.ViewModelLookup(typeof(CompileViewModel))!;
        if (ShapeBucket)
        {
            Context.InsertPage(shapeBucket, compile);
        }
        else
        {
            Context.RemovePage(shapeBucket);
        }
    }

    public override void UpdateViewModel()
    {
        InputFile = Context.CompileOption.InputFile;
        InputFormat = Context.CompileOption.InputFormat;
    }

    public override void UpdateContext()
    {
        Context.CompileOption.DumpDir = DumpDir;
        Context.CompileOption.DumpFlags = DumpFlagSelected.Aggregate(DumpFlags.None, (flag, sum) => flag | sum);
        Context.Target = Target;
        Context.CompileOption.PreProcess = Preprocess;
        Context.MixQuantize = MixQuantize;
        if (!Quantize)
        {
            Context.CompileOption.QuantizeOptions.ModelQuantMode = ModelQuantMode.NoQuant;
        }
    }

    public override List<string> CheckViewModel()
    {
        return new();
    }
}
