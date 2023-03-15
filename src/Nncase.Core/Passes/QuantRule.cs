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

namespace Nncase.Passes;

/// <summary>
/// Lower Rewrite rule.
/// </summary>
public abstract class QuantRule : RewriteRule<Pattern>
{
    /// <summary>
    /// Gets or sets context.
    /// NOTE the option will be set by SourceGenerator when the GetReplace called.
    /// </summary>
    public RunPassContext? Option { get; set; }

    /// <summary>
    /// Gets or sets the match result.
    /// NOTE the MatchResult will be set by SourceGenerator when the GetReplace called.
    /// </summary>
    public IMatchResult? MatchResult { get; set; }

    /// <summary>
    /// Gets whole expr be matched.
    /// </summary>
    public Expr Root => (Expr)MatchResult![Pattern];

    /// <summary>
    /// Gets get ModelQuantMode.
    /// </summary>
    public ModelQuantMode ModelQuantMode => CompileSession.CompileOptions.QuantizeOptions.ModelQuantMode;

    /// <summary>
    /// Gets get QuantType.
    /// </summary>
    public DataType QuantType => CompileSession.CompileOptions.QuantizeOptions.QuantType;

    /// <summary>
    /// Gets get WQuantType.
    /// </summary>
    public DataType WQuantType => CompileSession.CompileOptions.QuantizeOptions.WQuantType;

    /// <summary>
    /// Gets a value indicating whether get UseMixQuant flag.
    /// </summary>
    public bool UseMixQuant => CompileSession.CompileOptions.QuantizeOptions.BindQuantMethod;

    /// <summary>
    /// check the datatype is the quant type.
    /// </summary>
    public bool IsQuantType(DataType dt) => dt == DataTypes.Int8 || dt == DataTypes.UInt8;

    /// <summary>
    /// NOTE the Init will be set by SourceGenerator when the GetReplace called.
    /// </summary>
    public abstract void Init();
}
