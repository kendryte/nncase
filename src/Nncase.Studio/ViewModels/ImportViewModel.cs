// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.Input;

namespace Nncase.Studio.ViewModels;

public partial class ImportViewModel : ViewModelBase
{
    private string _inputFile = string.Empty;

    private string _inputFormat = string.Empty;

    public ImportViewModel(ViewModelContext context)
    {
        Context = context;
    }

    [RelayCommand]
    public async Task Import()
    {
        var path = await Context.OpenFile(PickerOptions.ImporterPickerOptions);
        if (path == null)
        {
            return;
        }

        if (path.Count == 0)
        {
            return;
        }

        var ext = Path.GetExtension(path[0]).Trim('.');
        if (!new[] { "onnx", "tflite", "ncnn" }.Contains(ext))
        {
            Context.OpenDialog($"Not Support {ext}");
            return;
        }

        _inputFormat = ext;
        _inputFile = path[0];
        Context.SwitchNext();
    }

    public override void UpdateContext()
    {
        Context.CompileOption.InputFile = _inputFile;
        Context.CompileOption.InputFormat = _inputFormat;
    }

    public override List<string> CheckViewModel()
    {
        var list = new List<string>();
        if (_inputFile == string.Empty)
        {
            list.Add("model path is empty");
        }
        else if (!File.Exists(_inputFile))
        {
            list.Add($"model path {_inputFile} Not Exist");
        }

        return list;
    }
}
