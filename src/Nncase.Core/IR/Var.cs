// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    /// <summary>
    /// Variable expression.
    /// </summary>
    public record Var(string Name, IRType TypeAnnotation) : Expr
    {
        public override IRType? CheckedType
        {
            get => TypeAnnotation;
            set { _checkedType = value == TypeAnnotation ? value : throw new InvalidOperationException("Can't Set CheckType != TypeAnnotation"); }
        }

        private static int _globalVarIndex = 0;

        /// <summary>
        /// Initializes a new instance of the <see cref="Var"/> class.
        /// </summary>
        /// <param name="typeAnnotation">Type annotation.</param>
        public Var(IRType typeAnnotation)
            : this($"var_{_globalVarIndex++}", typeAnnotation)
        {
        }

        public Var(string Name)
            : this(Name, AnyType.Default)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Var"/> class.
        /// </summary>
        public Var()
            : this($"var_{_globalVarIndex++}", AnyType.Default)
        {
        }

        /// <summary>
        /// get any var
        /// </summary>
        /// <param name="Name"></param>
        public static implicit operator Var(string Name) => new Var(Name, AnyType.Default);

        /// <summary>
        /// get scalar var
        /// </summary>
        /// <param name="Name"></param>
        public static Var Scalar(string Name, DataType Dtype) => new Var(Name, new TensorType(Dtype, Shape.Scalar));

        /// <summary>
        /// get handle var
        /// </summary>
        /// <param name="Name"></param>
        /// <param name="Dtype"></param>
        /// <param name="Scope"></param>
        /// <returns> var </returns>
        public static Var Handle(string Name, DataType Dtype, string Scope = "") => new Var(Name, new PointerType(Dtype, Scope));
    }
}
