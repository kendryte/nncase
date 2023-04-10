// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.Toolkit.HighPerformance.Helpers;
using Nncase;
using Nncase.IR;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestLSTMHelper
{
    [Fact]
    public void TestToLSTMDirection()
    {
        Assert.Equal(LSTMDirection.Forward, LSTMHelper.ToLSTMDirection("forward"));
        Assert.Equal(LSTMDirection.Reverse, LSTMHelper.ToLSTMDirection("reverse"));
        Assert.Equal(LSTMDirection.Bidirectional, LSTMHelper.ToLSTMDirection("bidirectional"));
        Assert.Throws<ArgumentOutOfRangeException>(() => LSTMHelper.ToLSTMDirection(string.Empty));
    }

    [Fact]
    public void TestToLSTMLayout()
    {
        Assert.Equal(LSTMLayout.Zero, LSTMHelper.ToLSTMLayout(0L));
        Assert.Equal(LSTMLayout.One, LSTMHelper.ToLSTMLayout(1L));
        Assert.Throws<ArgumentOutOfRangeException>(() => LSTMHelper.ToLSTMLayout(2L));
    }

    [Fact]
    public void TestLSTMDirectionToValue()
    {
        Assert.Equal("forward", LSTMHelper.LSTMDirectionToValue(LSTMDirection.Forward));
        Assert.Equal("reverse", LSTMHelper.LSTMDirectionToValue(LSTMDirection.Reverse));
        Assert.Equal("bidirectional", LSTMHelper.LSTMDirectionToValue(LSTMDirection.Bidirectional));
    }

    [Fact]
    public void TestLSTMLayoutToValue()
    {
        Assert.Equal(0, LSTMHelper.LSTMLayoutToValue(LSTMLayout.Zero));
        Assert.Equal(1, LSTMHelper.LSTMLayoutToValue(LSTMLayout.One));
    }
}
