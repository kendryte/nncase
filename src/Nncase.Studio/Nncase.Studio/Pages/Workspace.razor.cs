// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Components;

namespace Nncase.Studio.Pages;

public partial class Workspace
{
    [Parameter]
    [SupplyParameterFromQuery(Name = "path")]
    public string WorkingDirectory { get; set; } = default!;
}
