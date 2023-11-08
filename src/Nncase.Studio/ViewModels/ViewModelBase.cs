// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using ReactiveUI;

namespace Nncase.Studio.ViewModels;

public interface ISwitchable
{
    public List<string> CollectInvalidInput();

    public void UpdateUI();

    public void UpdateContext(ViewModelContext context);

    public bool IsVisible();
}

public class ViewModelBase : ObservableValidator
{
    protected ViewModelContext Context { get; set; } = new(new());

    public virtual List<string> CheckViewModel()
    {
        return new();
    }

    /// <summary>
    /// Update Data when switch page.
    /// </summary>
    public virtual void UpdateContext()
    {
    }

    public virtual void UpdateViewModel()
    {
    }

    public virtual bool IsVisible()
    {
        return true;
    }
}
