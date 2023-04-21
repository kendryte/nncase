// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components.Web;

namespace Nncase.Studio.Gtk.Blazor;

public sealed class RootComponentsCollection : ObservableCollection<RootComponent>, IJSComponentConfiguration
{
    public JSComponentConfigurationStore JSComponents { get; } = new();
}
