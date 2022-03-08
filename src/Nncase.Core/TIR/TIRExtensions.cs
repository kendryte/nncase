// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;


public static class TIRExtensions
{

    /// <summary>
    /// convert IEnumerable to tir Sequential.
    /// </summary>
    /// <param name="enumerable"> instance.</param>
    /// <returns> Sequential. </returns>
    public static Sequential ToSequential(this IEnumerable<Expr> enumerable) => new Sequential(new IRArrayList<Expr>(enumerable));
}