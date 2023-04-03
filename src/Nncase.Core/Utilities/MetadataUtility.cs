// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Immutable;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Fx = System.Func<Nncase.IR.Expr, Nncase.IR.Expr>;
using ParameterInfo = Nncase.IR.ParameterInfo;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Utilities;

/// <summary>
/// Metadata Utility.
/// </summary>
///
public static class MetadataUtility
{
    /// <summary>
    /// Inherit MetaData.
    /// </summary>
    ///
    public static Expr InheritMetaData(this Expr newCall, Expr oldCall)
    {
        if (oldCall.Metadata != null && oldCall.Metadata!.OutputNames != null)
        {
            newCall.Metadata.OutputNames = oldCall.Metadata.OutputNames;
        }

        return newCall;
    }
}
