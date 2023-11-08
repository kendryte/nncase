// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Avalonia.Rendering.Composition;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Hosting;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Mutators;
using Nncase.Quantization;
using Nncase.Runtime.Interop;
using Nncase.Studio.ViewModels;
using Nncase.Studio.Views;
using Nncase.Utilities;
using NumSharp;
using ReactiveUI;
using static Nncase.Studio.ViewModels.Helper;

namespace Nncase.Studio.ViewModels;

public enum PromptDialogLevel
{
    Normal,
    Error,
}

public partial class MainWindowViewModel : ViewModelBase
{
    public MainWindowViewModel()
    {
        InitViewModel();
        var host = Host.CreateDefaultBuilder()
            .ConfigureCompiler()
            .Build();
        CompilerServices.Configure(host.Services);
        ShowFilePicker = new();
        ShowFolderPicker = new();
        ShowPromptDialog = new();
        var tmpVar = new Var("testVar", new TensorType(DataTypes.Float32, new[] { 1, 3, 24, 24 }));
        MainParamStr = new ObservableCollection<string>(new[] { VarToString(tmpVar) });
        DumpDir = Path.Join(Directory.GetCurrentDirectory(), "nncase_dump");
        KmodelPath = Path.Join(DumpDir, "test.kmodel");
        ResultDir = Path.Join(DumpDir, "nncase_result");
        ExportQuantSchemePath = Path.Join(DumpDir, "QuantScheme.json");
    }

    public List<string> Validate()
    {
        // todo: validate
        var option = CompileOptionViewModel.Validate();
        if (CompileOptionViewModel.Preprocess)
        {
            var preprocess = PreprocessViewModel.Validate();
            option = option.Concat(preprocess).ToList();
        }

        if (CompileOptionViewModel.Quantize)
        {
            var quantize = QuantizeViewModel.Validate();
            option = option.Concat(quantize).ToList();
        }

        return option;
    }

    private CompileOptions GetCompileOption()
    {
        var options = new CompileOptions();
        CompileOptionViewModel.UpdateCompileOption(options);
        if (CompileOptionViewModel.Quantize)
        {
            QuantizeViewModel.UpdateCompileOption(options);
        }

        if (CompileOptionViewModel.Preprocess)
        {
            PreprocessViewModel.UpdateCompileOption(options);
        }

        return options;
    }

    private void InitViewModel()
    {
        var context = new ViewModelContext(this);
        ImportViewModel = new ImportViewModel(context);
        CompileOptionViewModel = new CompileOptionViewModel(context);
        PreprocessViewModel = new PreprocessViewModel(context);
        QuantizeViewModel = new QuantizeViewModel(context);
        CompileViewModel = new CompileViewModel(context);
        // SimulateInputViewModel = new SimulateInputViewModel(context);
        SimulateViewModel = new SimulateViewModel(context);
        ContentViewModelList = new ObservableCollection<ViewModelBase>(
            new ViewModelBase[]
            {
                ImportViewModel,
                CompileOptionViewModel,
                CompileViewModel,
                SimulateViewModel,
            });

        ShowQuantize();
        ShowPreprocess();
        UpdateContentViewModel();
    }

    private void UpdateContentViewModel()
    {
        PageMaxIndex = ContentViewModelList.Count - 1;
        ContentViewModel = ContentViewModelList[PageIndex];
        Title = ContentViewModel.GetType().Name.Split("ViewModel")[0];
        PageIndexString = $"{PageIndex + 1} / {PageCount}";
        IsLast = PageIndex == PageMaxIndex;
    }

    public async Task ShowDialog(string prompt, PromptDialogLevel level = PromptDialogLevel.Error)
    {
        // todo: dialog level
        await ShowPromptDialog.Handle((prompt, level));
    }
}

public sealed class SelfInputCalibrationDatasetProvider : ICalibrationDatasetProvider
{
    private readonly int _count = 1;

    private readonly IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> _samples;

    public SelfInputCalibrationDatasetProvider(IReadOnlyDictionary<Var, IValue> sample)
    {
        _samples = new[] { sample }.ToAsyncEnumerable();
    }

    public int? Count => _count;

    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples => _samples;
}
