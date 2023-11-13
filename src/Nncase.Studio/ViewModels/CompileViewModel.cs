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
using Nncase.Quantization;
using Nncase.Studio.Views;
using ReactiveUI;

namespace Nncase.Studio.ViewModels;

public partial class CompileViewModel : ViewModelBase
{
    private CancellationTokenSource _cts = new();

    [ObservableProperty]
    private int _progressBarMax = 9;

    [ObservableProperty]
    private int _progressBarValue;

    [ObservableProperty]
    private string _kmodelPath = "test.kmodel";

    [ObservableProperty]
    private string _p = string.Empty;

    public CompileViewModel(ViewModelContext context)
    {
        Context = context;
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
        if (!Directory.Exists(options.DumpDir))
        {
            Directory.CreateDirectory(options.DumpDir);
        }

        ITarget target;
        try
        {
            target = CompilerServices.GetTarget(Context.Target);
        }
        catch (Exception e)
        {
            Context.OpenDialog(e.Message, PromptDialogLevel.Error);
            return;
        }

        var compileSession = CompileSession.Create(target, options);
        var compiler = compileSession.Compiler;
        var module = await compiler.ImportModuleAsync(options.InputFormat, options.InputFile, options.IsBenchmarkOnly);
        Context.Entry = (Function)module.Entry!;
        if (options.QuantizeOptions.ModelQuantMode != ModelQuantMode.NoQuant)
        {
            var calib = ((QuantizeViewModel)Context.ViewModelLookup(typeof(QuantizeViewModel))).LoadCalibFiles();
            if (calib == null)
            {
                return;
            }

            options.QuantizeOptions.CalibrationDataset = calib;
        }

        _cts = new();

        ProgressBarMax = 9;
        var progress = new Progress<int>(percent =>
        {
            ProgressBarValue = percent;
        });

        try
        {
            await Task.Run(async () =>
            {
                await compiler.CompileWithReport(progress, _cts.Token);
            }).ContinueWith(_ => Task.CompletedTask, _cts.Token);
        }
        catch (Exception)
        {
            Context.OpenDialog("Compile has been cancel");
            ProgressBarValue = 0;
            return;
        }

        using (var os = File.OpenWrite(KmodelPath))
        {
            compiler.Gencode(os);
        }

        Context.SwitchNext();
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
