// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using MethodBoundaryAspect.Fody.Attributes;
using Xunit;

namespace Nncase.Tests.TestFixture;

/// <summary>
/// Automatically call <see cref="TestClassBase.SetupTestMethod(bool, string?, string?)"/>.
/// </summary>
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false)]
public sealed class AutoSetupTestMethodAttribute : OnMethodBoundaryAspect
{
    /// <summary>
    /// Gets or sets a value indicating whether to initialize <see cref="TestClassBase.CompileSession"/>.
    /// </summary>
    public bool InitSession { get; set; }

    /// <summary>
    /// Gets or sets target name, null to use <see cref="TestClassBase.DefaultTargetName"/>.
    /// </summary>
    public string? TargetName { get; set; }

    /// <inheritdoc/>
    public override void OnEntry(MethodExecutionArgs arg)
    {
        if (Attribute.IsDefined(arg.Method, typeof(FactAttribute), false)
            || Attribute.IsDefined(arg.Method, typeof(TheoryAttribute), false))
        {
            var testClass = (TestClassBase)arg.Instance;
            testClass.SetupTestMethod(InitSession, TargetName, arg.Method.Name);
        }
    }
}
