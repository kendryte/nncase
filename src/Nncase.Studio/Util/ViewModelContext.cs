// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using System.Threading.Tasks;
using Avalonia.Platform.Storage;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Studio.ViewModels;

namespace Nncase.Studio;

public class ViewModelContext
{
    private readonly MainWindowViewModel _mainWindowView;

    public ViewModelContext(MainWindowViewModel windowViewModel)
    {
        _mainWindowView = windowViewModel;
    }

    public CompileOptions CompileOption { get; } = new();

    public NavigatorViewModel? Navigator { get; set; }

    public ViewModelBase[] ViewModelBases { get; set; } = Array.Empty<ViewModelBase>();

    public bool MixQuantize { get; set; }

    public bool UseQuantize { get; set; }

    public bool CustomPreprocessMode { get; set; }

    public bool EnableShapeBucket
    {
        get { return CompileOption.ShapeBucketOptions.Enable; }
        set { CompileOption.ShapeBucketOptions.Enable = value; }
    }

    public string KmodelPath { get; set; } = string.Empty;

    public string Target { get; set; } = "cpu";

    public Function? Entry { get; set; }

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

    public void SwitchToPage(Type page)
    {
        var viewModel = ViewModelLookup(page);
        Navigator?.SwitchToPage(viewModel);
    }

    public void SwitchNext()
    {
        // todo: switch check is validate
        Navigator?.SwitchNext();
    }

    public ViewModelBase ViewModelLookup(Type type)
    {
        foreach (var viewModelBase in ViewModelBases)
        {
            if (viewModelBase.GetType() == type)
            {
                return viewModelBase;
            }
        }

        throw new InvalidOperationException($"{type} Not Found");
    }

    public string[] CheckViewModel()
    {
        return Navigator!.ContentViewModelList.SelectMany(x => x.CheckViewModel()).ToArray();
    }
}
