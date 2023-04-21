// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Studio.Services;

public interface IFolderPicker
{
    Task<string?> PickFolderAsync(string title, string? defaultPath = null);
}
