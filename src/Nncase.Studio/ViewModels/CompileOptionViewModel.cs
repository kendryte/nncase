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
        Context = context;
    }

    public void UpdateCompileOption(CompileOptions options)
    {
        options.DumpFlags = DumpFlagSelected.Aggregate(DumpFlags.None, (flag, sum) => flag & sum);
        // options.PreProcess = Preprocess;
        // options.InputFile = InputFile;
        // options.InputFormat = InputFormat;
        // options.DumpDir = DumpDir;
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
        var folder = await Context.OpenFolder(PickerOptions.FolderPickerOpenOptions);
        if (folder != string.Empty)
        {
            DumpDir = folder;
        }
    }



        // var i = ContentViewModelList.IndexOf(PreprocessViewModel);
        // if (CompileOptionViewModel.Preprocess)
        // {
        //     if (i == -1)
        //     {
        //         var optionIndex = ContentViewModelList.IndexOf(CompileOptionViewModel);
        //         // insert after OptionView
        //         ContentViewModelList.Insert(optionIndex + 1, PreprocessViewModel);
        //     }
        // }
        // else
        // {
        //     if (i != -1)
        //     {
        //         ContentViewModelList.Remove(PreprocessViewModel);
        //     }
        // }

        // var i = ContentViewModelList.IndexOf(QuantizeViewModel);
        // if (CompileOptionViewModel.Quantize)
        // {
        //     var compileIndex = ContentViewModelList.IndexOf(CompileViewModel);
        //
        //     // insert before CompileView
        //     ContentViewModelList.Insert(compileIndex, QuantizeViewModel);
        // }
        // else
        // {
        //     if (i != -1)
        //     {
        //         ContentViewModelList.Remove(QuantizeViewModel);
        //     }
        // }


        [RelayCommand]
        public void UseMixQuantize()
        {
            // QuantizeViewModel.MixQuantize = CompileOptionViewModel.MixQuantize;
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
}
