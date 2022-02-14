// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Pattern
{
    public abstract partial record ExprPattern
    {
        public static implicit operator ExprPattern(byte value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(ushort value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(uint value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(ulong value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(sbyte value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(short value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(int value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(long value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(Half value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(float value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(double value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(BFloat16 value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(bool value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(Tensor value) => (TensorConstPattern)value;

        public static implicit operator ExprPattern(int[] span) => Const.FromSpan<int>(span);

        public static implicit operator ExprPattern(float[] span) => Const.FromSpan<float>(span);
    }
}
