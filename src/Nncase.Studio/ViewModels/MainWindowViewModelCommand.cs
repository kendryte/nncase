using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
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
using Nncase.Studio.Views;
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

    public bool IsVisible();
}

public class ViewModelContext
{
    private MainWindowViewModel _mainWindowView;

    public NavigatorViewModel? Navigator;

    public ViewModelBase[] ViewModelBases;

    public CompileOptions CompileOption = new();

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

    public CompileOptions GetCompileOption()
    {
        return new();
    }

    public void InsertPage(ViewModelBase page, ViewModelBase pagePosition, int offset = 0)
    {
        Navigator?.InsertPageAfter(page, pagePosition, offset);
    }

    public void RemovePage(ViewModelBase page)
    {
        Navigator?.RemovePage(page);
    }

    public void SwitchNext()
    {
        Navigator?.SwitchNext();
    }

    public ViewModelBase? ViewModelLookup(Type type)
    {
        foreach (var viewModelBase in ViewModelBases)
        {
            if (viewModelBase.GetType() == type)
            {
                return viewModelBase;
            }
        }

        return null;
    }
}


// property
public partial class MainWindowViewModel
{
    [ObservableProperty]
    private string _title;

    [ObservableProperty]
    private ViewModelBase _contentViewModel;

    public Interaction<FilePickerOpenOptions, List<string>> ShowFilePicker { get; }

    public Interaction<FolderPickerOpenOptions, string> ShowFolderPicker { get; }

    public Interaction<(string, PromptDialogLevel), Unit> ShowPromptDialog { get; }
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
