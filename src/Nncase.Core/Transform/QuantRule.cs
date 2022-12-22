// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.PatternMatch;
using Nncase.Quantization;

namespace Nncase.Transform;

/// <summary>
/// Lower Rewrite rule.
/// </summary>
public abstract class QuantRule : RewriteRule<Pattern>
{
    /// <summary>
    /// NOTE the option will be set by SourceGenerator when the GetReplace called.
    /// </summary>
    public RunPassOptions Option = null!;

    /// <summary>
    /// the match result
    /// NOTE the MatchResult will be set by SourceGenerator when the GetReplace called.
    /// </summary>
    public IMatchResult MatchResult = null!;

    /// <summary>
    /// Gets whole expr be matched.
    /// </summary>
    public Expr Root => (Expr)MatchResult[Pattern];

    /// <summary>
    /// Gets get ModelQuantMode.
    /// </summary>
    public ModelQuantMode ModelQuantMode => Option.CompileOptions.ModelQuantMode;

    /// <summary>
    /// Gets get QuantType.
    /// </summary>
    public DataType QuantType => Option.CompileOptions.QuantType;

    /// <summary>
    /// check the datatype is the quant type.
    /// </summary>
    /// <param name="dt"></param>
    /// <returns></returns>
    public bool IsQuantType(DataType dt) => dt == DataTypes.Int8 || dt == DataTypes.UInt8;

    /// <summary>
    /// Gets get WQuantType.
    /// </summary>
    public DataType WQuantType => Option.CompileOptions.WQuantType;

    /// <summary>
    /// Gets a value indicating whether get UseMixQuant flag.
    /// </summary>
    public bool UseMixQuant => Option.CompileOptions.QuantizeOptions.BindQuantMethod;

    /// <summary>
    /// NOTE the Init will be set by SourceGenerator when the GetReplace called.
    /// </summary>
    public abstract void Init();
}
