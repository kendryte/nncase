// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.Input;

namespace Nncase.Studio.ViewModels;

public partial class ImportViewModel : ViewModelBase
{
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
            Context.OpenDialog($"Not Support {ext}");
            return;
        }

        // CompileOptionViewModel.InputFile = path[0];
        // CompileOptionViewModel.InputFormat = ext;

        Context.SwitchNext();
    }

}
