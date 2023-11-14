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
using static Nncase.Studio.ViewModels.DataUtil;

namespace Nncase.Studio.ViewModels;

public enum PromptDialogLevel
{
    Normal,
    Error,
}

public partial class MainWindowViewModel : WindowViewModelBase
{
    [ObservableProperty]
    private string _title;

    [ObservableProperty]
    private ViewModelBase _contentViewModel;

    public MainWindowViewModel()
    {
        var host = Host.CreateDefaultBuilder()
            .ConfigureCompiler()
            .Build();
        CompilerServices.Configure(host.Services);
        Context = new ViewModelContext(this);
        var importViewModel = new ImportViewModel(Context);
        var compileOptionViewModel = new CompileOptionViewModel(Context);
        var preprocessViewModel = new PreprocessViewModel(Context);
        var quantizeViewModel = new QuantizeViewModel(Context);
        var shapeBucketViewModel = new ShapeBucketViewModel(Context);
        var compileViewModel = new CompileViewModel(Context);

        // var SimulateInputViewModel = new SimulateInputViewModel(Context);
        var simulateViewModel = new SimulateViewModel(Context);
        var contentViewModelList = new ObservableCollection<ViewModelBase>(
            new ViewModelBase[]
            {
                importViewModel,
                compileOptionViewModel,
                preprocessViewModel,
                quantizeViewModel,
                shapeBucketViewModel,
                compileViewModel,
                simulateViewModel,
            });

        Title = string.Empty;
        ContentViewModel = contentViewModelList.First();
        Context.ViewModelBases = contentViewModelList.ToArray();
        NavigatorViewModelValue = new NavigatorViewModel(contentViewModelList, Update);
        Context.Navigator = NavigatorViewModelValue;
        NavigatorViewModelValue.UpdateContentViewModel();
    }

    public Interaction<FilePickerOpenOptions, List<string>> ShowFilePicker { get; } = new();

    public Interaction<FolderPickerOpenOptions, string> ShowFolderPicker { get; } = new();

    public Interaction<(string Message, PromptDialogLevel Level), Unit> ShowPromptDialog { get; } = new();

    public NavigatorViewModel NavigatorViewModelValue { get; set; }

    protected ViewModelContext Context { get; set; }

    public void Update(ViewModelBase contenViewModel)
    {
        ContentViewModel = contenViewModel;
        Title = ContentViewModel.GetType().Name.Split("ViewModel")[0];
    }

    public async Task ShowDialog(string prompt, PromptDialogLevel level = PromptDialogLevel.Error)
    {
        await ShowPromptDialog.Handle((prompt, level));
    }
}
