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
    private bool _mixQuantize = false;

    [ObservableProperty]
    private string _inputFormat = string.Empty;

    [ObservableProperty]
    private string _target;

    public ObservableCollection<DumpFlags> DumpFlagSelected { get; set; } = new();

    public ObservableCollection<DumpFlags> DumpFlagsList { get; set; }

    public ObservableCollection<string> TargetList { get; set; }

    public CompileOptionViewModel(ViewModelContext context)
    {
        // skip None
        DumpFlagsList = new ObservableCollection<DumpFlags>(System.Enum.GetValues<DumpFlags>().Skip(1).ToList());
        TargetList = new ObservableCollection<string>(new[] { "cpu", "k230" });

        // todo: k230 should set dll path
        _target = TargetList[0];
        DumpDir = Path.Join(Directory.GetCurrentDirectory(), "nncase_dump");
    }

    public void UpdateCompileOption(CompileOptions options)
    {
        options.DumpFlags = DumpFlagSelected.Aggregate(DumpFlags.None, (flag, sum) => flag & sum);
        options.PreProcess = Preprocess;
        options.InputFile = InputFile;
        options.InputFormat = InputFormat;
        options.DumpDir = DumpDir;
    }

    public List<string> Validate()
    {
        return new();

        // todo: 什么时候做import
        // InputFile
    }

    [RelayCommand]
    public async Task SetDumpDir()
    {
        var folder = await ShowFolderPicker.Handle(PickerOptions.FolderPickerOpenOptions);
        if (folder != string.Empty)
        {
            DumpDir = folder;
        }
    }
}
