// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.IR;
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

    [RelayCommand]
    public async Task Compile()
    {
        // todo: validate
        // var info = Validate();
        // if (info.Count != 0)
        // {
            // Context.OpenDialog($"Error List:\n{string.Join("\n", info)}");
            // return;
        // }

        var options = Context.GetCompileOption();
        // todo: target
        var target = CompilerServices.GetTarget("");
        var compileSession = CompileSession.Create(target, options);
        var compiler = compileSession.Compiler;
        if (!File.Exists(options.InputFile))
        {
            Context.OpenDialog($"File Not Exist {options.InputFile}");
            return;
        }

        var _module = await compiler.ImportModuleAsync(options.InputFormat, options.InputFile, options.IsBenchmarkOnly);

        // todo:
        // update progress bar
        var progress = new Progress<int>(percent => { });

        // await Task.Run(() => );

        await compiler.CompileAsync().ContinueWith(_ => Task.CompletedTask, _token);

        // // todo: kmodel 默认加version info
        // using (var os = File.OpenWrite(CompileViewModel.KmodelPath))
        // {
        //     compiler.Gencode(os);
        // }

        Context.SwitchNext();
        var main = (Function)_module.Entry!;
        // MainParamStr = new ObservableCollection<string>(main.Parameters.ToArray().Select(VarToString));
        Context.OpenDialog("Compile Finish", PromptDialogLevel.Normal);
    }

}
