// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.IR;
using NumSharp;

namespace Nncase.Studio.ViewModels;

public partial class SimulateViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _resultDir = string.Empty;

    [ObservableProperty]
    private string _kmodelPath = string.Empty;

    [ObservableProperty] private string _status = "未运行";

    [ObservableProperty]
    private ObservableCollection<Tensor> _runtimeInput = new();

    [ObservableProperty]
    private ObservableCollection<string> _inputPath = new();

    [ObservableProperty]
    private ObservableCollection<string> _mainParamStr = new();

    [ObservableProperty]
    private ObservableCollection<string> _inputTypeStr = new();

    public SimulateViewModel(ViewModelContext context)
    {
        this.Context = context;
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
        }
        catch (Exception e)
        {
            Context.OpenDialog(e.Message);
            return;
        }

        UpdateRuntimeInputUI(input, inputFiles);
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
    public void Simulate()
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

        try
        {
            // todo: 字体问题？？？

            // todo: 通过kmodel检查input，是否应当支持直接跑kmodel，如果支持的话那就必须在function的接口添加input信息的地方
            // todo: log能否重定向, compile and simulate， simulate如何log和进度
            using (var interp = Runtime.Interop.RTInterpreter.Create())
            {
                var kmodel = File.ReadAllBytes(KmodelPath);
                interp.SetDumpRoot(Context.CompileOption.DumpDir);
                interp.LoadModel(kmodel);
                var entry = interp.Entry!;
                var rtInputs = RuntimeInput.Select(Runtime.Interop.RTTensor.FromTensor).ToArray();

                Status = "Running";
                var result = entry.Invoke(rtInputs).ToValue().AsTensors();

                var list = result
                    .Select(t =>
                        np.frombuffer(t.BytesBuffer.ToArray(), t.ElementType.CLRType)
                            .reshape(t.Shape.ToValueArray()))
                    .ToArray();

                for (int i = 0; i < list.Length; i++)
                {
                    np.save(Path.Join(ResultDir, $"nncase_result_{i}.npy"), list[i]);
                }

                Status = "Finish";
                Context.OpenDialog($"Simulate Finish, result in {ResultDir}", PromptDialogLevel.Normal);
            }
            return;
        }
        catch (DllNotFoundException e)
        {
            Context.OpenDialog("libNncase.Native.so not found");
            return;
        }
        catch (Exception e)
        {
            var msg = ExceptionMessageProcess(e);
            Context.OpenDialog(msg);
            return;
        }
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
                errcStr = "invalid model indentifier"; break;
            case 0x02:
                errcStr = "invalid model checksum"; break;
            case 0x03:
                errcStr = "invalid model version"; break;
            case 0x04:
                errcStr = "runtime not found"; break;
            case 0x05:
                errcStr = "datatype mismatch"; break;
            case 0x06:
                errcStr = "shape mismatch"; break;
            case 0x07:
                errcStr = "invalid memory location"; break;
            case 0x08:
                errcStr = "runtime register not found"; break;
            case 0x0100:
                errcStr = "stackvm illegal instruction"; break;
            case 0x0101:
                errcStr = "stackvm illegal target"; break;
            case 0x0102:
                errcStr = "stackvm stack overflow"; break;
            case 0x0103:
                errcStr = "stackvm stack underflow"; break;
            case 0x0104:
                errcStr = "stackvm unknow custom call"; break;
            case 0x0105:
                errcStr = "stackvm duplicate custom call"; break;
            case 0x0200:
                errcStr = "nnil illegal instruction"; break;
            default:
                errcStr = $"Unknown Status code: {errc}";
                break;
        }

        return errcStr;
    }

    // private bool CheckInput(out Task simulate)
    // {
    //     if (Context.Entry == null)
    //     {
    //         Context.OpenDialog("Should Import Model first");
    //         {
    //             simulate = Task.CompletedTask;
    //             return true;
    //         }
    //     }
    //
    //     var paramList = Context.Entry!.Parameters.ToArray();
    //     foreach ((var tensor, var param) in RuntimeInput.Zip(paramList))
    //     {
    //         var tt = (TensorType)param.TypeAnnotation;
    //         if (tensor.ElementType != tt.DType)
    //         {
    //             Context.OpenDialog($"{param.Name} input datatype mismatch");
    //             {
    //                 simulate = Task.CompletedTask;
    //                 return true;
    //             }
    //         }
    //
    //         if (tt.Shape.Count != tensor.Shape.Count || tt.Shape.Zip(tensor.Shape)
    //                 .Any(pair => pair.First.IsFixed && pair.First != pair.Second))
    //         {
    //             Context.OpenDialog($"{param.Name} input shape mismatch");
    //             {
    //                 simulate = Task.CompletedTask;
    //                 return true;
    //             }
    //         }
    //     }
    //
    //     return false;
    // }

    public override void UpdateViewModel()
    {
        if (ResultDir == string.Empty)
        {
            ResultDir = Context.CompileOption.DumpDir;
        }

        KmodelPath = Context.KmodelPath;
    }

    private void UpdateRuntimeInputUI(Tensor[] input, string[] inputFiles)
    {
        RuntimeInput = new ObservableCollection<Tensor>(input);
        InputPath = new ObservableCollection<string>(inputFiles);
        InputTypeStr = new ObservableCollection<string>(RuntimeInput
            .Select(x => DataUtil.TensorTypeToString(new TensorType(x.ElementType, x.Shape))).ToList());
    }
}
