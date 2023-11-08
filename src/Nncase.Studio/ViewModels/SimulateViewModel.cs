// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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

        var (inputFiles, input) = Helper.ReadMultiInputs(path);
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
        }

        var paramList = Context.Entry.Parameters.ToArray();
        foreach ((var tensor, var param) in RuntimeInput.Zip(paramList))
        {
            var tt = (TensorType)param.TypeAnnotation;
            if (tensor.ElementType != tt.DType)
            {
                Context.OpenDialog($"{param.Name} input datatype mismatch");
                return;
            }

            if (tt.Shape.Count != tensor.Shape.Count || tt.Shape.Zip(tensor.Shape)
                    .Any(pair => pair.First.IsFixed && pair.First != pair.Second))
            {
                Context.OpenDialog($"{param.Name} input shape mismatch");
                return;
            }
        }

        using (var interp = Runtime.Interop.RTInterpreter.Create())
        {
            var kmodel = File.ReadAllBytes(KmodelPath);
            interp.SetDumpRoot(Context.DumpDir);
            interp.LoadModel(kmodel);
            var entry = interp.Entry!;
            var rtInputs = RuntimeInput.Select(Runtime.Interop.RTTensor.FromTensor).ToArray();
            var result = entry.Invoke(rtInputs).ToValue().AsTensors();

            var list = result.Select(t => np.ndarray(new NumSharp.Shape(t.Shape.ToValueArray()), t.ElementType.CLRType,
                t.BytesBuffer.ToArray())).ToArray();

            // todo: output file name, collect in compiler
            foreach (var ndArray in list)
            {
                np.save(Path.Join(ResultDir, "dir.npy"), ndArray);
            }

            // BinFileUtil.WriteBinOutputs(result, Path.Join(DumpDir, "output"));
            Context.OpenDialog("Simulate Finish", PromptDialogLevel.Normal);
        }

        // todo: open kmodel in explorer
        // todo: show log
    }

    private void UpdateRuntimeInputUI(Tensor[] input, string[] inputFiles)
    {
        RuntimeInput = new ObservableCollection<Tensor>(input);
        InputPath = new ObservableCollection<string>(inputFiles);
        InputTypeStr = new ObservableCollection<string>(RuntimeInput
            .Select(x => Helper.TensorTypeToString(new TensorType(x.ElementType, x.Shape))).ToList());
    }
}
