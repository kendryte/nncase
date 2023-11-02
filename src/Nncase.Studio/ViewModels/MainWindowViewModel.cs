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
using Microsoft.Extensions.Hosting;
using Nncase.Diagnostics;
using Nncase.Passes.Mutators;
using Nncase.Quantization;
using Nncase.Studio.ViewModels;
using Nncase.Utilities;
using ReactiveUI;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.IR;
using Nncase.Runtime.Interop;
using Nncase.Studio.Views;
using NumSharp;

namespace Nncase.Studio.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{
    public static FilePickerOpenOptions DataPickerOptions = new FilePickerOpenOptions
    {
        Title = "Open Input File",
        AllowMultiple = true,
        FileTypeFilter = new FilePickerFileType[]{ new("npy") { Patterns = new[]{"*.npy"}}}
    };

    public static FilePickerOpenOptions JsonPickerOptions = new FilePickerOpenOptions
    {
        Title = "Open Json File",
        AllowMultiple = true,
        FileTypeFilter = new FilePickerFileType[]{ new("json") { Patterns = new[]{"*.json"}}}
    };

    public static FilePickerOpenOptions ImporterPickerOptions = new FilePickerOpenOptions
    {
        Title = "Open Model File",
        AllowMultiple = false,
        FileTypeFilter = new FilePickerFileType[]
        {
            new("model") { Patterns = new[]{"*.tflite", "*.onnx", "*.ncnn"}},
        }
    };

    public Interaction<FilePickerOpenOptions, List<string>> ShowFilePicker { get; }

    public Interaction<string, Unit> ShowPromptDialog { get; }

    [ObservableProperty] private ViewModelBase _contentViewModel;

    public ObservableCollection<ViewModelBase> ContentViewModelList { get; set; } = new();

    [ObservableProperty] private int _pageIndex = 0;

    public int PageCount => ContentViewModelList.Count;

    [ObservableProperty]
    private int _PageMaxIndex;

    public MainWindowViewModel()
    {
        InitViewModel();
        var host = Host.CreateDefaultBuilder()
            .ConfigureCompiler()
            .Build();
        CompilerServices.Configure(host.Services);
        // ShowDialog = new Interaction<OptionViewModel, OptionViewModel?>();
        ShowFilePicker = new();
        ShowPromptDialog = new();
        var tmpVar = new Var("testVar", new TensorType(DataTypes.Float32, new[] { 1, 3, 24, 24 }));
        MainParamStr = new ObservableCollection<string>(new[]
        {
            VarToString(tmpVar)
        });
        token = cts.Token;
    }

    private string TensorTypeToString(TensorType tt)
    {
        return $"{tt.DType} {tt.Shape}";
    }

    private string VarToString(Var var)
    {
        var tt = (TensorType)var.TypeAnnotation;
        return $"{var.Name} {TensorTypeToString(tt)}";
    }

    [ObservableProperty]
    private bool _isLast;

    [ObservableProperty]
    private string _pageIndexString;

    [ObservableProperty]
    private ImportViewModel _importViewModel;
    [ObservableProperty]
    private CompileOptionViewModel _compileOptionViewModel;
    [ObservableProperty]
    private QuantizeViewModel _quantizeViewModel;
    [ObservableProperty]
    private PreprocessViewModel _preprocessViewModel;
    [ObservableProperty]
    private CompileViewModel _compileViewModel;
    [ObservableProperty]
    private SimulateInputViewModel _simulateInputViewModel;
    [ObservableProperty]
    private SimulateViewModel _simulateViewModel;
    public string KmodelPath
    {
        get
        {
            var path = _compileViewModel.KmodelPath!;
            if (path.Contains("/", StringComparison.Ordinal) || path.Contains("\\", StringComparison.Ordinal))
            {
                // is full path
                return path;
            }
            else
            {
                // is file name
                return Path.Join(DumpDir, path);
            }
        }
    }

    [ObservableProperty]
    private string _title;

    public string DumpDir => _compileOptionViewModel.DumpDir;

    private IRModule _module;

    [RelayCommand]
    public void SwitchPrev()
    {
        // PageIndex -= 1;
        if (PageIndex != 0)
        {
            PageIndex -= 1;
        }

        UpdateContentViewModel();
    }

    [RelayCommand]
    public void SwitchNext()
    {
        // todo: update to next, and set current background color

        // PageIndex += 1;
        if (PageIndex != PageMaxIndex)
        {
            PageIndex += 1;
        }

        UpdateContentViewModel();
    }

    [RelayCommand]
    public void ShowPreprocess()
    {
        var i = ContentViewModelList.IndexOf(_preprocessViewModel);
        if (_compileOptionViewModel.Preprocess)
        {
            if (i == -1)
            {
                var optionIndex = ContentViewModelList.IndexOf(_compileOptionViewModel);
                // insert after OptionView
                ContentViewModelList.Insert(optionIndex + 1, _preprocessViewModel);
            }
        }
        else
        {
            if (i != -1)
            {
                ContentViewModelList.Remove(_preprocessViewModel);
            }
        }
    }

    [RelayCommand]
    public void ShowQuantize()
    {
        var i = ContentViewModelList.IndexOf(_quantizeViewModel);
        if (_compileOptionViewModel.Quantize)
        {
            var compileIndex = ContentViewModelList.IndexOf(_compileViewModel);
            // insert before CompileView
            ContentViewModelList.Insert(compileIndex, _quantizeViewModel);
        }
        else
        {
            if (i != -1)
            {
                ContentViewModelList.Remove(_quantizeViewModel);
            }
        }
    }

    [RelayCommand]
    public async void CancelCompile()
    {
        cts.Cancel();
    }

    [ObservableProperty]
    private ObservableCollection<Tensor> _runtimeInput = new();

    [ObservableProperty] private ObservableCollection<string> _inputPath = new();

    [ObservableProperty] private ObservableCollection<string> _mainParamStr = new();

    [ObservableProperty] private ObservableCollection<string> _inputTypeStr = new();

    private CancellationTokenSource cts = new();
    private CancellationToken token;

    private List<Tensor> ReadInput(string[] file)
    {
        return file.Select(f =>
        {
            var ext = Path.GetExtension(f);
            if (ext != ".npy")
            {
                throw new NotImplementedException();
            }

            var tensor = np.load(f);
            return Tensor.FromBytes(new TensorType(ToDataType(tensor.dtype), tensor.shape), tensor.ToByteArray());
        }).ToList();
    }

    private DataType ToDataType(Type type)
    {
        if (type == typeof(byte))
        {
            return DataTypes.UInt8;
        }
        if (type == typeof(sbyte))
        {
            return DataTypes.Int8;
        }
        if (type == typeof(ushort))
        {
            return DataTypes.UInt16;
        }
        if (type == typeof(short))
        {
            return DataTypes.Int16;
        }
        if (type == typeof(int))
        {
            return DataTypes.Int32;
        }
        if (type == typeof(uint))
        {
            return DataTypes.UInt32;
        }
        if (type == typeof(long))
        {
            return DataTypes.Int64;
        }
        if (type == typeof(ulong))
        {
            return DataTypes.UInt64;
        }
        if (type == typeof(float))
        {
            return DataTypes.Float32;
        }
        if (type == typeof(double))
        {
            return DataTypes.Float64;
        }
        if (type == typeof(bool))
        {
            return DataTypes.Boolean;
        }

        // todo: bf16, float16
        throw new NotImplementedException();
    }

    [RelayCommand]
    public async void SetRuntimeInput()
    {
        var path = await ShowFilePicker.Handle(DataPickerOptions);
        if (path.Count == 0)
        {
            return;
        }

        var (inputFiles, input) = ReadMultiInputs(path);
        UpdateRuntimeInputUI(input, inputFiles);
    }

    private (string[], Tensor[]) ReadMultiInputs(List<string> path)
    {
        var inputFiles = path.Count == 1 && Directory.Exists(path[0])
            ? Directory.GetFiles(path[0])
            : path.ToArray();
        var input = ReadInput(inputFiles).ToArray();
        return (inputFiles, input);
    }

    private void UpdateRuntimeInputUI(Tensor[] input, string[] inputFiles)
    {
        RuntimeInput = new ObservableCollection<Tensor>(input);
        InputPath = new ObservableCollection<string>(inputFiles);
        InputTypeStr = new ObservableCollection<string>(RuntimeInput
            .Select(x => TensorTypeToString(new TensorType(x.ElementType, x.Shape))).ToList());
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

    [RelayCommand]
    public async void Compile()
    {
        var info = Validate();
        if (info.Count != 0)
        {
            await ShowDialog($"Error List:\n{string.Join("\n", info)}");
            return;
        }

        var options = GetCompileOption();
        var target = CompilerServices.GetTarget(_compileOptionViewModel.Target);
        var compileSession = CompileSession.Create(target, options);
        var compiler = compileSession.Compiler;
        if (!File.Exists(options.InputFile))
        {
            await ShowDialog($"File Not Exist {options.InputFile}");
            return;
        }

        _module = await compiler.ImportModuleAsync(options.InputFormat, options.InputFile, options.IsBenchmarkOnly);

        // todo:
        // update progress bar
        var progress = new Progress<int>(percent => { });
        // await Task.Run(() => );



        await compiler.CompileAsync().ContinueWith(async _ =>
        {
            // todo: show this
            // todo: and switch next
            SwitchNext();
            var main = (Function)_module.Entry;
            MainParamStr = new ObservableCollection<string>(main.Parameters.ToArray().Select(VarToString));
            Console.WriteLine(KmodelPath);
            await ShowDialog("Compile Finish");
        }, token);

        // todo: kmodel 默认加version info
        // todo: kmodel name and path
        // todo: compile option save to file
        using (var os = File.OpenWrite(KmodelPath))
        {
            compiler.Gencode(os);
        }

        // open a dialog
        // todo: generate kmodel finish, kmodel path: xxx
    }

    private CompileOptions GetCompileOption()
    {
        var options = new CompileOptions();
        _compileOptionViewModel.UpdateCompileOption(options);
        if (_compileOptionViewModel.Quantize)
        {
            _quantizeViewModel.UpdateCompileOption(options);
        }

        if (_compileOptionViewModel.Preprocess)
        {
            _preprocessViewModel.UpdateCompileOption(options);
        }

        return options;
    }

    [RelayCommand]
    public async void Import()
    {
        var path = await ShowFilePicker.Handle(ImporterPickerOptions);
        if (path == null)
        {
            Console.WriteLine("FileNotSelected");
            return;
        }

        if (path.Count == 0)
        {
            return;
        }

        var ext = Path.GetExtension(path[0]).Trim('.');
        if (!new[] { "onnx", "tflite", "ncnn" }.Contains(ext))
        {
            await ShowDialog($"Not Support {ext}");
            return;
        }

        _compileOptionViewModel.InputFile = path[0];
        _compileOptionViewModel.InputFormat = ext;
        SwitchNext();
    }

    private async Task ShowDialog(string prompt)
    {
        await ShowPromptDialog.Handle(prompt);
    }

    [RelayCommand]
    public async void SelectQuantScheme()
    {
        var path = await ShowFilePicker.Handle(JsonPickerOptions);
        if (path.Count == 0)
        {
            return;
        }

        QuantizeViewModel.QuantSchemePath = path[0];
        // todo: path or content?
    }

    [RelayCommand]
    public async void SelectCalibrationDataSet()
    {
        var path = await ShowFilePicker.Handle(DataPickerOptions);
        if (path.Count == 0)
        {
            return;
        }

        // todo: fix this??
        var (inputFiles, input) = ReadMultiInputs(path);
        // todo: 直接import
        // QuantizeViewModel.QuantizeOptions.CalibrationDataset = new SelfInputCalibrationDatasetProvider(samples);
    }

    [RelayCommand]
    public async void Simulate()
    {
        if (!File.Exists(KmodelPath))
        {
            await ShowDialog("Kmodel Not Exist");
            return;
        }

        if (RuntimeInput.Count == 0)
        {
            await ShowDialog("Not Set Input");
        }

        var paramList = ((Function)_module.Entry).Parameters.ToArray();
        foreach ((var tensor, var param) in RuntimeInput.Zip(paramList))
        {
            var tt = (TensorType)param.TypeAnnotation;
            if (tensor.ElementType != tt.DType)
            {
                await ShowDialog($"{param.Name} input datatype mismatch");
                return;
            }

            if (tt.Shape.Count != tensor.Shape.Count || tt.Shape.Zip(tensor.Shape).Any(pair => pair.First.IsFixed && pair.First != pair.Second))
            {
                await ShowDialog($"{param.Name} input shape mismatch");
                return;
            }
        }

        using (var interp = Runtime.Interop.RTInterpreter.Create())
        {
            var kmodel = File.ReadAllBytes(KmodelPath);
            interp.SetDumpRoot(DumpDir);
            interp.LoadModel(kmodel);
            var entry = interp.Entry!;
            var rtInputs = RuntimeInput.Select(Runtime.Interop.RTTensor.FromTensor).ToArray();
            var result = entry.Invoke(rtInputs).ToValue().AsTensors();
            BinFileUtil.WriteBinOutputs(result, Path.Join(DumpDir, "output"));
            await ShowDialog("Simulate Finish");
            // todo: dump result
        }
        // todo: open kmodel in explorer
        // todo: show log
    }

    private void InitViewModel()
    {
        _importViewModel = new ImportViewModel();
        _compileOptionViewModel = new CompileOptionViewModel();
        _preprocessViewModel = new PreprocessViewModel();
        _quantizeViewModel = new QuantizeViewModel();
        _compileViewModel = new CompileViewModel();
        _simulateInputViewModel = new SimulateInputViewModel();
        _simulateViewModel = new SimulateViewModel();
        ContentViewModelList = new ObservableCollection<ViewModelBase>(
            (new ViewModelBase[]
            {
                _importViewModel,
                _compileOptionViewModel,
                _compileViewModel,
                _simulateInputViewModel,
                _simulateViewModel,
            }));

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
