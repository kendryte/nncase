// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Passes;

internal sealed class AnalyzerManager : IAnalyzerManager
{
    private readonly Dictionary<Type, IAnalyzerFactory> _analyzerFactories;

    public AnalyzerManager(IAnalyzerFactory[] analyzerFactories)
    {
        _analyzerFactories = analyzerFactories.ToDictionary(x => x.ResultType);
    }

    public IAnalyzerFactory GetFactory(Type resultType) => _analyzerFactories[resultType];
}
