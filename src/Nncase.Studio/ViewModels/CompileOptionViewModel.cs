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

public class PreprocessConfig
{
    private int[] _inputShape;

    private InputType _inputTypeValue;

    private int[] _inputTypeString;

    public string InputLayout { get; set; } = "NCHW";

    public string OutputLayout { get; set; } = "NCHW";

    public string ModelLayout { get; set; } = "NCHW";

    public bool SwapRB { get; set; }

    public float RangeMin { get; set; }

    public float RangeMax { get; set; }

    public float LetterBoxValue { get; set; }

    public float[] Mean { get; set; }

    public float[] Std { get; set; }
}

// todo: button 字体？？
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

    [ObservableProperty]
    public string _preprocessMode = CustomMode;

    public static string CustomMode = "自定义";

    public CompileOptionViewModel(ViewModelContext context)
    {
        // skip None
        DumpFlagsList = new ObservableCollection<DumpFlags>(System.Enum.GetValues<DumpFlags>().Skip(1).ToList());
        TargetList = new ObservableCollection<string>(new[] { "cpu", "k230" });

        _target = TargetList[0];
        DumpDir = Path.Join(Directory.GetCurrentDirectory(), "nncase_dump");
        Context = context;
        var list = new[] { CustomMode };
        PreprocessModeList = new(list);
    }

    public ObservableCollection<DumpFlags> DumpFlagSelected { get; set; } = new();

    public ObservableCollection<DumpFlags> DumpFlagsList { get; set; }

    public ObservableCollection<string> TargetList { get; set; }

    public ObservableCollection<string> PreprocessModeList { get; set; }

    [RelayCommand]
    public async Task SetDumpDir()
    {
        var folder = await Context.OpenFolder(PickerOptions.FolderPickerOpenOptions);
        if (folder != string.Empty)
        {
            DumpDir = folder;
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
        Context.EnableShapeBucket = ShapeBucket;
        Context.UseQuantize = Quantize;
        Context.CustomPreprocessMode = PreprocessMode == CustomMode;
        if (Quantize == false)
        {
            Context.CompileOption.QuantizeOptions.ModelQuantMode = ModelQuantMode.NoQuant;
        }
    }

    // todo simulate生成随机数
    // todo 每个页面valid以后再到下一页
    // todo validate那边修复间距和字体
    // todo system prompt dialog
    // todo dialog中的路径可以被复制

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
