// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Shapes;

namespace Nncase.IR.F;

/// <summary>
/// Shapes functional helper.
/// </summary>
public static class Shapes
{
    public static Call AsTensor(Dimension input) => new Call(new AsTensor(), input);

    public static Call AsTensor(Shape input) => new Call(new AsTensor(), input);
}
