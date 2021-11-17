// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Pattern
{
    public abstract partial record ExprPattern
    {
        public static implicit operator ExprPattern(byte value) => (ConstPattern)value;

        public static implicit operator ExprPattern(ushort value) => (ConstPattern)value;

        public static implicit operator ExprPattern(uint value) => (ConstPattern)value;

        public static implicit operator ExprPattern(ulong value) => (ConstPattern)value;

        public static implicit operator ExprPattern(sbyte value) => (ConstPattern)value;

        public static implicit operator ExprPattern(short value) => (ConstPattern)value;

        public static implicit operator ExprPattern(int value) => (ConstPattern)value;

        public static implicit operator ExprPattern(long value) => (ConstPattern)value;

        public static implicit operator ExprPattern(Half value) => (ConstPattern)value;

        public static implicit operator ExprPattern(float value) => (ConstPattern)value;

        public static implicit operator ExprPattern(double value) => (ConstPattern)value;

        public static implicit operator ExprPattern(BFloat16 value) => (ConstPattern)value;

        public static implicit operator ExprPattern(bool value) => (ConstPattern)value;
    }
}
