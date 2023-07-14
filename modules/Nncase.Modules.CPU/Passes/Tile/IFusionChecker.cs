// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Passes.Tile;

internal interface IFusionChecker
{
    /// <summary>
    /// 检查fusion是否可以正常执行.
    /// </summary>
    /// <param name="fusion">fusion.</param>
    /// <param name="passOptions">passOptions.</param>
    /// <returns>.</returns>
    public bool Check(Fusion fusion, RunPassContext passOptions);

    /// <summary>
    /// 通常当check过一个fusion之后, 可以cache部分的内容, 此时通过convert复用.
    /// </summary>
    /// <param name="passOptions">passOptions.</param>
    /// <returns>.</returns>
    public TIR.PrimFunction Convert(RunPassContext passOptions);
}
