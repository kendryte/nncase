// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using Nncase.Studio.ViewModels;

namespace Nncase.Studio;

public class ViewLocator : IDataTemplate
{
    public Control Build(object? data)
    {
        if (data == null)
        {
            return new TextBlock { Text = "Not Found" };
        }

        var name = data!.GetType().FullName!.Replace("ViewModel", "View", StringComparison.Ordinal);
        var type = Type.GetType(name);

        if (type != null)
        {
            return (Control)Activator.CreateInstance(type)!;
        }

        return new TextBlock { Text = "Not Found: " + name };
    }

    public bool Match(object? data)
    {
        return data is ViewModelBase;
    }
}
