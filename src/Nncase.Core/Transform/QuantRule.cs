// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;

namespace Nncase.Transform;

/// <summary>
/// Lower Rewrite rule.
/// </summary>
public abstract class QuantRule : RewriteRule<Pattern>
{
    public RunPassOptions Option;
    
    public Expr Root;
    
    public bool IsQuantType(DataType dt) => dt == DataTypes.Int8 || dt == DataTypes.UInt8;
    
    public bool UsePTQ => Option.CompileOptions.UsePTQ;
    
    public DataType QuantType => Option.CompileOptions.QuantType;

    public QuantMode QuantMode => Option.CompileOptions.QuantMode;

    public abstract void Init();

}