// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using ReactiveUI;

namespace Nncase.Studio.ViewModels;

// todo: Observable Object and ReactObject
public class ViewModelBase : ObservableValidator
{
    protected ViewModelContext Context;
}
