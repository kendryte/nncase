﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Distributed;

namespace Nncase.IR.F;

public partial class Distributed
{
    public static Call Boxing(Expr input, IRType type)
    {
        return new Call(new Boxing(type), input);
    }

    public static Call ForceBoxing(Expr input, DistributedType type)
    {
        return new Call(new ForceBoxing(type), input);
    }
}
