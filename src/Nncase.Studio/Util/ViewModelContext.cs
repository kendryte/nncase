// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Threading.Tasks;
using Avalonia.Platform.Storage;
using Nncase.IR;
using Nncase.Quantization;
using Nncase.Studio.Util;
using Nncase.Studio.ViewModels;

namespace Nncase.Studio;

public class ViewModelContext
{
    private readonly MainWindowViewModel _mainWindowView;

    public ViewModelContext(MainWindowViewModel windowViewModel)
    {
        _mainWindowView = windowViewModel;
    }

    public CompileConfig CompileConfig { get; set; } = new();

    public bool CustomPreprocessMode => CompileConfig.PreprocessMode == PreprocessMode.Custom;

    public NavigatorViewModel? Navigator { get; set; }

    public ViewModelBase[] ViewModelBases { get; set; } = Array.Empty<ViewModelBase>();

    public Var[] Params { get; set; } = Array.Empty<Var>();

    public async Task<List<string>> OpenFile(FilePickerOpenOptions options)
    {
        return await _mainWindowView.ShowOpenFilePicker.Handle(options);
    }

    public async Task<string> SaveFile(FilePickerSaveOptions options)
    {
        return await _mainWindowView.ShowSaveFilePicker.Handle(options);
    }

    public async Task<string> OpenFolder(FolderPickerOpenOptions options)
    {
        return await _mainWindowView.ShowFolderPicker.Handle(options);
    }

    public void OpenDialog(string prompt, PromptDialogLevel level = PromptDialogLevel.Error)
    {
        _mainWindowView.ShowDialog(prompt, level);
    }

    public void SwitchToPage(System.Type page)
    {
        var viewModel = ViewModelLookup(page);
        Navigator?.SwitchToPage(viewModel);
    }

    public void SwitchNext()
    {
        Navigator?.SwitchNext();
    }

    public ViewModelBase ViewModelLookup(System.Type type)
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

    public CompileConfig ExportConfig()
    {
        return CompileConfig;
    }

    public void ImportConfig(CompileConfig conf)
    {
        CompileConfig = conf;
    }
}
