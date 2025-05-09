// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.NN;

namespace Nncase.Runtime.Interop;

public sealed class RTPagedAttentionScheduler : RTObject, IAttentionScheduler
{
    internal RTPagedAttentionScheduler()
        : base(IntPtr.Zero)
    {
    }

    internal RTPagedAttentionScheduler(IntPtr handle, bool addRef = false)
            : base(handle)
    {
        if (addRef)
        {
            Native.ObjectAddRef(handle);
        }
    }

    public static RTPagedAttentionScheduler Create(RTPagedAttentionConfig config, int numBlocks, int maxModelLen)
    {
        throw new NotSupportedException();

        // Native.PagedAttentionSchedulerCreate(config, numBlocks, maxModelLen, out var scheduler).ThrowIfFailed();
        // return scheduler;
    }

    public static RTPagedAttentionScheduler Create(PagedAttentionConfig config, int numBlocks, int maxModelLen)
    {
        var rtConfig = RTPagedAttentionConfig.FromConfig(config);
        return Create(rtConfig, numBlocks, maxModelLen);
    }

    public IAttentionKVCache Schedule(Tensor<long> sessionIds, Tensor<long> tokenCounts)
    {
        throw new NotSupportedException();

        // Native.PagedAttentionSchedulerSchedule(this, RTTensor.FromTensor(sessionIds), RTTensor.FromTensor(tokenCounts), out var cache).ThrowIfFailed();
        // return cache;
    }
}
