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

    // public CompileOptions CompileOption
    // {
    //     get { return CompileConfig.CompileOption; }
    //     set { CompileConfig.CompileOption = value; }
    // }
    //
    // public bool MixQuantize
    // {
    //     get { return CompileConfig.MixQuantize; }
    //     set { CompileConfig.MixQuantize = value; }
    // }
    //
    // public bool UseQuantize
    // {
    //     get { return CompileConfig.UseQuantize; }
    //     set { CompileConfig.UseQuantize = value; }
    // }
    public bool CustomPreprocessMode { get; set; }

    // public string KmodelPath
    // {
    //     get { return CompileConfig.KmodelPath; }
    //     set { CompileConfig.KmodelPath = value; }
    // }
    //
    // public string Target
    // {
    //     get { return CompileConfig.Target; }
    //     set { CompileConfig.Target = value; }
    // }
    public NavigatorViewModel? Navigator { get; set; }

    public ViewModelBase[] ViewModelBases { get; set; } = Array.Empty<ViewModelBase>();

    public Function? Entry { get; set; }

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

    public void SwitchToPage(Type page)
    {
        var viewModel = ViewModelLookup(page);
        Navigator?.SwitchToPage(viewModel);
    }

    public void SwitchNext()
    {
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

    public CompileConfig ExportConfig()
    {
        return CompileConfig;
    }

    public void ImportConfig(CompileConfig conf)
    {
        CompileConfig = conf;
    }
}
