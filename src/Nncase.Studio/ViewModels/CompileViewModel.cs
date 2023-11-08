// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.Studio.Views;
using ReactiveUI;

namespace Nncase.Studio.ViewModels;

public partial class CompileViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _kmodelPath = "test.kmodel";

    private readonly CancellationToken _token;
    private readonly CancellationTokenSource _cts = new();

    public CompileViewModel(ViewModelContext context)
    {
        _token = _cts.Token;
    }

    [RelayCommand]
    public Task CancelCompile()
    {
        _cts.Cancel();
        return Task.CompletedTask;
    }

    public async Task Compile(ICompiler compiler)
    {
        // todo:
        // update progress bar
        var progress = new Progress<int>(percent => { });

        // await Task.Run(() => );

        await compiler.CompileAsync().ContinueWith(_ => Task.CompletedTask, _token);
    }
}
