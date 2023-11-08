using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Reactive.Linq;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Platform.Storage;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Nncase.IR;
using NumSharp;
using ReactiveUI;
using static Nncase.Studio.ViewModels.Helper;
using static Nncase.Studio.PickerOptions;
namespace Nncase.Studio.ViewModels;

public interface ISwitchable
{
    public List<string> CollectInvalidInput();

    public void UpdateUI();

    public void UpdateContext(ViewModelContext context);
}

public class ViewModelContext
{
    private MainWindowViewModel _mainWindowView;

    public Function Entry;

    public string KmodelPath;

    public string DumpDir;

    public ViewModelContext(MainWindowViewModel windowViewModel)
    {
        _mainWindowView = windowViewModel;
    }

    public async Task<List<string>> OpenFile(FilePickerOpenOptions options)
    {
        return await _mainWindowView.ShowFilePicker.Handle(options);
    }

    public async Task<string> OpenFolder(FolderPickerOpenOptions options)
    {
        return await _mainWindowView.ShowFolderPicker.Handle(options);
    }

    public async void OpenDialog(string prompt, PromptDialogLevel level = PromptDialogLevel.Error)
    {
        await _mainWindowView.ShowDialog(prompt, level);
    }
}

// command
public partial class MainWindowViewModel
{
    // todo: FilePicker and Dialog
    [RelayCommand]
    public void SwitchPrev()
    {
        if (PageIndex != 0)
        {
            PageIndex -= 1;
        }

        UpdateContentViewModel();
    }

    [RelayCommand]
    public void SwitchNext()
    {
        if (PageIndex != PageMaxIndex)
        {
            PageIndex += 1;
        }

        UpdateContentViewModel();
    }

    [RelayCommand]
    public void ShowPreprocess()
    {
        var i = ContentViewModelList.IndexOf(PreprocessViewModel);
        if (CompileOptionViewModel.Preprocess)
        {
            if (i == -1)
            {
                var optionIndex = ContentViewModelList.IndexOf(CompileOptionViewModel);
                // insert after OptionView
                ContentViewModelList.Insert(optionIndex + 1, PreprocessViewModel);
            }
        }
        else
        {
            if (i != -1)
            {
                ContentViewModelList.Remove(PreprocessViewModel);
            }
        }
    }

    [RelayCommand]
    public void ShowQuantize()
    {
        var i = ContentViewModelList.IndexOf(QuantizeViewModel);
        if (CompileOptionViewModel.Quantize)
        {
            var compileIndex = ContentViewModelList.IndexOf(CompileViewModel);

            // insert before CompileView
            ContentViewModelList.Insert(compileIndex, QuantizeViewModel);
        }
        else
        {
            if (i != -1)
            {
                ContentViewModelList.Remove(QuantizeViewModel);
            }
        }
    }


    [RelayCommand]
    public async Task Import()
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
            ShowDialog($"Not Support {ext}");
            return;
        }

        CompileOptionViewModel.InputFile = path[0];
        CompileOptionViewModel.InputFormat = ext;

        SwitchNext();
    }

    [RelayCommand]
    public async Task Compile()
    {
        var info = Validate();
        if (info.Count != 0)
        {
            ShowDialog($"Error List:\n{string.Join("\n", info)}");
            return;
        }

        var options = GetCompileOption();
        var target = CompilerServices.GetTarget(CompileOptionViewModel.Target);
        var compileSession = CompileSession.Create(target, options);
        var compiler = compileSession.Compiler;
        if (!File.Exists(options.InputFile))
        {
            ShowDialog($"File Not Exist {options.InputFile}");
            return;
        }

        _module = await compiler.ImportModuleAsync(options.InputFormat, options.InputFile, options.IsBenchmarkOnly);

        await CompileViewModel.Compile(compiler);

        // todo: kmodel 默认加version info
        using (var os = File.OpenWrite(KmodelPath))
        {
            compiler.Gencode(os);
        }

        SwitchNext();
        var main = (Function)_module.Entry!;
        // MainParamStr = new ObservableCollection<string>(main.Parameters.ToArray().Select(VarToString));
        ShowDialog("Compile Finish", PromptDialogLevel.Normal);
    }

    [RelayCommand]
    public void UseMixQuantize()
    {
        QuantizeViewModel.MixQuantize = CompileOptionViewModel.MixQuantize;
    }
}

// property
public partial class MainWindowViewModel
{
    [ObservableProperty]
    private ViewModelBase _contentViewModel;
    public int PageCount => ContentViewModelList.Count;

    [ObservableProperty]
    private int _pageIndex = 0;

    [ObservableProperty]
    private int _pageMaxIndex;

    [ObservableProperty]
    private bool _isLast;

    [ObservableProperty]
    private string _pageIndexString;

    public ObservableCollection<ViewModelBase> ContentViewModelList { get; set; } = new();






    public Interaction<FilePickerOpenOptions, List<string>> ShowFilePicker { get; }

    public Interaction<FolderPickerOpenOptions, string> ShowFolderPicker { get; }

    public Interaction<(string, PromptDialogLevel), Unit> ShowPromptDialog { get; }

    private IRModule _module;



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



    [ObservableProperty]
    private string _kmodelPath;

    [ObservableProperty]
    private string _title;
}

public static class Helper
{
    public static string TensorTypeToString(TensorType tt)
    {
        return $"{tt.DType} {tt.Shape}";
    }

    public static string VarToString(Var var)
    {
        var tt = (TensorType)var.TypeAnnotation;
        return $"{var.Name} {TensorTypeToString(tt)}";
    }

    public static (string[], Tensor[]) ReadMultiInputs(List<string> path)
    {
        var inputFiles = path.Count == 1 && Directory.Exists(path[0])
            ? Directory.GetFiles(path[0])
            : path.ToArray();
        var input = ReadInput(inputFiles).ToArray();
        return (inputFiles, input);
    }

    public static List<Tensor> ReadInput(string[] file)
    {
        return file
            .Where(f => Path.GetExtension(f) == ".npy")
            .Select(f =>
            {
                var tensor = np.load(f);
                return Tensor.FromBytes(new TensorType(ToDataType(tensor.dtype), tensor.shape), tensor.ToByteArray());
            }).ToList();
    }

    // todo: not support datatype error
    public static DataType ToDataType(Type type)
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
}
