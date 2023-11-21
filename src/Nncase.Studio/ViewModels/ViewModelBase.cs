// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using Nncase.Studio.Util;
using ReactiveUI;

namespace Nncase.Studio.ViewModels;

public class WindowViewModelBase : ObservableValidator
{
}

public class ViewModelBase : ObservableValidator
{
    private ViewModelContext? _context;

    protected ViewModelContext Context
    {
        get { return _context!; }
        set { _context = value; }
    }

    public void UpdateContext()
    {
        UpdateConfig(Context.CompileConfig);
    }

    public void UpdateViewModel()
    {
        UpdateViewModelCore(Context.CompileConfig);
    }

    public virtual List<string> CheckViewModel()
    {
        return new();
    }

    public virtual void UpdateConfig(CompileConfig config)
    {
    }

    public virtual void UpdateViewModelCore(CompileConfig config)
    {
    }

    public virtual bool IsVisible()
    {
        return true;
    }
}
