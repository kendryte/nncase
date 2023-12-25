// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.IR;
using Nncase.Studio.Util;
using NumSharp;

namespace Nncase.Studio.ViewModels;

public partial class SimulateViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _resultDir = string.Empty;

    [ObservableProperty]
    private string _kmodelPath = string.Empty;

    [ObservableProperty]
    private string _status = "未运行";

    [ObservableProperty]
    private ObservableCollection<Tensor> _runtimeInput = new();

    [ObservableProperty]
    private ObservableCollection<string> _inputPath = new();

    [ObservableProperty]
    private ObservableCollection<string> _mainParamStr = new();

    [ObservableProperty]
    private ObservableCollection<string> _inputTypeStr = new();

    [ObservableProperty]
    private bool _runSimulate;

    public SimulateViewModel(ViewModelContext context)
    {
        Context = context;
    }

    [RelayCommand]
    public async Task SetRuntimeInput()
    {
        var path = await Context.OpenFile(PickerOptions.DataPickerOptions);
        if (path.Count == 0)
        {
            return;
        }

        Tensor[] input;
        string[] inputFiles;
        try
        {
            (inputFiles, input) = DataUtil.ReadMultiInputs(path);
            if (CanBeSort(inputFiles))
            {
                var pairList = inputFiles.Zip(input)
                    .OrderBy(pair => int.Parse(Path.GetFileName(pair.First).Split("_")[0]));
                inputFiles = pairList.Select(x => x.First).ToArray();
                input = pairList.Select(x => x.Second).ToArray();
            }
        }
        catch (Exception e)
        {
            Context.OpenDialog(e.Message);
            return;
        }

        UpdateRuntimeInputUI(input, inputFiles);

        bool CanBeSort(string[] inputFilePathList)
        {
            var fileNames = inputFilePathList.Select(Path.GetFileName).ToArray();
            var canBeSort = fileNames.All(x =>
                x!.Contains("_", StringComparison.Ordinal) && int.TryParse(x.Split("_")[0], out int _));
            return canBeSort;
        }
    }

    [RelayCommand]
    public async Task SetResultDir()
    {
        var folder = await Context.OpenFolder(PickerOptions.FolderPickerOpenOptions);
        if (folder != string.Empty)
        {
            ResultDir = folder;
        }
    }

    [RelayCommand]
    public async Task Simulate()
    {
        if (!File.Exists(KmodelPath))
        {
            Context.OpenDialog("Kmodel Not Exist");
            return;
        }

        if (RuntimeInput.Count == 0)
        {
            Context.OpenDialog("Not Set Input");
            return;
        }

        if (!CheckInput())
        {
            return;
        }

        var cts = new CancellationTokenSource();

        try
        {
            using (var interp = Runtime.Interop.RTInterpreter.Create())
            {
                var kmodel = File.ReadAllBytes(KmodelPath);
                interp.SetDumpRoot(Context.CompileConfig.CompileOption.DumpDir);
                interp.LoadModel(kmodel);
                var entry = interp.Entry!;
                var rtInputs = RuntimeInput.Select(Runtime.Interop.RTTensor.FromTensor).ToArray();

                var start = System.DateTime.Now;
                Status = "Running";
                RunSimulate = true;

                var task = Task.Run(
                    () =>
                    {
                        UpdateTime(cts, start);
                    },
                    cts.Token);
                await Task.Run(() =>
                {
                    var result = entry.Invoke(rtInputs).ToValue().AsTensors();
                    SaveResult(result);
                }).ContinueWith(t => t, cts.Token);

                SimulateFinish(cts);
                Context.OpenDialog($"Simulate Finish, spent time:{Status}\n result in {ResultDir}", PromptDialogLevel.Normal);
                Status = "Finish";
            }
        }
        catch (DllNotFoundException)
        {
            Context.OpenDialog("libNncase.Native.so not found");
        }
        catch (Exception e)
        {
            var msg = ExceptionMessageProcess(e);
            Context.OpenDialog(msg);
        }
        finally
        {
            SimulateFinish(cts);
        }
    }

    public override void UpdateViewModelCore(CompileConfig config)
    {
        ResultDir = config.ResultDir;
        KmodelPath = config.KmodelPath;
        InputPath = new(config.InputPathList);
    }

    public override void UpdateConfig(CompileConfig config)
    {
        config.ResultDir = ResultDir;
        config.InputPathList = InputPath.ToArray();
    }

    private bool CheckInput()
    {
        if (Context.Params.Length == 0)
        {
            return true;
        }

        var paramList = Context.Params!;
        foreach ((var tensor, var param) in RuntimeInput.Zip(paramList))
        {
            var tt = (TensorType)param.TypeAnnotation;
            if (tensor.ElementType != tt.DType)
            {
                Context.OpenDialog($"{param.Name} input datatype mismatch");
                {
                    return false;
                }
            }

            if (tt.Shape.Count != tensor.Shape.Count || tt.Shape.Zip(tensor.Shape)
                    .Any(pair => pair.First.IsFixed && pair.First != pair.Second))
            {
                Context.OpenDialog($"{param.Name} input shape mismatch");
                {
                    return false;
                }
            }
        }

        return true;
    }

    [RelayCommand]
    private async Task SetKmodelPath()
    {
        var kmodel = await Context.OpenFile(PickerOptions.KmodelPickerOptions);
        if (kmodel.Count == 0)
        {
            return;
        }

        KmodelPath = kmodel[0];
    }

    private void SaveResult(Tensor[] result)
    {
        var list = result
            .Select(t =>
                np.frombuffer(t.BytesBuffer.ToArray(), t.ElementType.CLRType)
                    .reshape(t.Shape.ToValueArray()))
            .ToArray();

        for (int i = 0; i < list.Length; i++)
        {
            np.save(Path.Join(ResultDir, $"nncase_result_{i}.npy"), list[i]);
        }
    }

    private void UpdateTime(CancellationTokenSource cts, DateTime start)
    {
        while (true)
        {
            Thread.Sleep(20);
            if (cts.Token.IsCancellationRequested)
            {
                return;
            }

            var now = System.DateTime.Now;
            var timeStr = (now - start).ToString();
            Status = timeStr.Substring(3, timeStr.Length - 8);
        }
    }

    private void SimulateFinish(CancellationTokenSource cts)
    {
        cts.Cancel();
        RunSimulate = false;
    }

    private string ExceptionMessageProcess(Exception exception)
    {
        var msg = exception.Message;
        if (msg.Contains("Status code", StringComparison.Ordinal))
        {
            if (int.TryParse(msg.Split(":")[1].Trim(), out var errc))
            {
                return ErrcToString(errc);
            }

            return exception.ToString();
        }

        return exception.ToString();
    }

    private string ErrcToString(int errc)
    {
        string errcStr;
        switch (-errc)
        {
            case 0x01:
                errcStr = "invalid model indentifier";
                break;
            case 0x02:
                errcStr = "invalid model checksum";
                break;
            case 0x03:
                errcStr = "invalid model version";
                break;
            case 0x04:
                errcStr = "runtime not found";
                break;
            case 0x05:
                errcStr = "datatype mismatch";
                break;
            case 0x06:
                errcStr = "shape mismatch";
                break;
            case 0x07:
                errcStr = "invalid memory location";
                break;
            case 0x08:
                errcStr = "runtime register not found";
                break;
            case 0x0100:
                errcStr = "stackvm illegal instruction";
                break;
            case 0x0101:
                errcStr = "stackvm illegal target";
                break;
            case 0x0102:
                errcStr = "stackvm stack overflow";
                break;
            case 0x0103:
                errcStr = "stackvm stack underflow";
                break;
            case 0x0104:
                errcStr = "stackvm unknow custom call";
                break;
            case 0x0105:
                errcStr = "stackvm duplicate custom call";
                break;
            case 0x0200:
                errcStr = "nnil illegal instruction";
                break;
            default:
                errcStr = $"Unknown Status code: {errc}";
                break;
        }

        return errcStr;
    }

    private void UpdateRuntimeInputUI(Tensor[] input, string[] inputFiles)
    {
        RuntimeInput = new ObservableCollection<Tensor>(input);
        InputPath = new ObservableCollection<string>(inputFiles);
        InputTypeStr = new ObservableCollection<string>(RuntimeInput
            .Select(x => DataUtil.TensorTypeToString(new TensorType(x.ElementType, x.Shape))).ToList());
    }
}
