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
    private readonly CancellationTokenSource _cts = new();

    private readonly CancellationToken _token;

    [ObservableProperty]
    private string _kmodelPath = "test.kmodel";

    public CompileViewModel(ViewModelContext context)
    {
        _token = _cts.Token;
        this.Context = context;
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
        var info = Context.CheckViewModel();
        if (info.Length != 0)
        {
            Context.OpenDialog($"Error List:\n{string.Join("\n", info)}");
            return;
        }

        var options = Context.CompileOption;
        var compileSession = Context.CreateCompileSession();
        var compiler = compileSession.Compiler;
        var module = await compiler.ImportModuleAsync(options.InputFormat, options.InputFile, options.IsBenchmarkOnly);

        // todo:
        // update progress bar
        var progress = new Progress<int>(percent => { });

        // await Task.Run(() => );
        await compiler.CompileAsync().ContinueWith(_ => Task.CompletedTask, _token);

        // // todo: kmodel 默认加version info
        using (var os = File.OpenWrite(KmodelPath))
        {
            compiler.Gencode(os);
        }

        Context.SwitchNext();
        var main = (Function)module.Entry!;

        // MainParamStr = new ObservableCollection<string>(main.Parameters.ToArray().Select(VarToString));
        Context.OpenDialog("Compile Finish", PromptDialogLevel.Normal);
    }

    public override void UpdateContext()
    {
        Context.KmodelPath = KmodelPath;
    }

    public override void UpdateViewModel()
    {
        KmodelPath = Context.KmodelPath;
        if (KmodelPath == string.Empty)
        {
            KmodelPath = Path.Join(Context.CompileOption.DumpDir, "test.kmodel");
        }
    }
}
