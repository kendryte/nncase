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

    public static RTPagedAttentionScheduler Create(int max_model_len)
    {
        Native.PagedAttentionSchedulerCreate(max_model_len, out var scheduler).ThrowIfFailed();
        return scheduler;
    }

    public void Initialize(RTPagedAttentionConfig config, int num_blocks)
    {
        Native.PagedAttentionSchedulerInitialize(this, config, num_blocks).ThrowIfFailed();
    }

    public IAttentionKVCache Schedule(Tensor<long> session_ids, Tensor<long> tokens_count)
    {
        throw new NotSupportedException();
        // Native.PagedAttentionSchedulerSchedule(this, session_ids, tokens_count, out var cache).ThrowIfFailed();
        // return cache;
    }
}
