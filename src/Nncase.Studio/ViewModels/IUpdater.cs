// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Studio.ViewModels;

public interface IUpdater
{
    public void UpdateCompileOption(CompileOptions options);

    public bool Validate();
}