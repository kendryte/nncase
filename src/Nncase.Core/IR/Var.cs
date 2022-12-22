// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR
{
    internal static partial class NameAlloc
    {
        private static readonly Dictionary<string, int> VarMaps = new();

        /// <summary>
        /// get unique var name, avoid the confilct name.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static string GetUniqueVarName(string name)
        {
            if (!VarMaps.TryGetValue(name, out var count))
            {
                count = 0;
                VarMaps.Add(name, count);
                return name;
            }

            while (VarMaps.ContainsKey(name + ++count))
            {
            }
            name = name + count;
            VarMaps[name] = 0;
            return name;
        }

        /// <summary>
        /// add the exits name into dict.
        /// </summary>
        /// <param name="name"></param>
        public static void AddName(string name)
        {
            if (!VarMaps.ContainsKey(name))
            {
                VarMaps[name] = 0;
            }
        }
    }

    /// <summary>
    /// Variable expression.
    /// </summary>
    public record Var : Expr
    {
        private static int _globalVarIndex;

        /// <summary>
        /// Initializes a new instance of the <see cref="Var"/> class.
        /// ctor.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="typeAnnotation"></param>
        public Var(string name, IRType typeAnnotation)
        {
            Name = name;
            TypeAnnotation = typeAnnotation;
            CheckedType = TypeAnnotation;
        }

        /// <summary>
        /// Gets get the global var index.
        /// </summary>
        private int GlobalVarIndex => _globalVarIndex;

        /// <summary>
        /// Gets name.
        /// </summary>
        public string Name { get; init; }

        /// <summary>
        /// Gets typeAnnotation.
        /// </summary>
        public IRType TypeAnnotation { get; init; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Var"/> class.
        /// </summary>
        /// <param name="typeAnnotation">Type annotation.</param>
        public Var(IRType typeAnnotation)
            : this($"var_{_globalVarIndex++}", typeAnnotation)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Var"/> class.
        /// <see cref="Var"/>.
        /// </summary>
        /// <param name="name"></param>
        public Var(string name)
            : this(name, AnyType.Default)
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
        /// get any var.
        /// </summary>
        /// <param name="name"></param>
        public static implicit operator Var(string name) => new Var(name, AnyType.Default);

        /// <summary>
        /// get scalar var.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="dtype"></param>
        /// <returns></returns>
        public static Var Scalar(string name, DataType dtype) => new Var(name, new TensorType(dtype, Shape.Scalar));

        /// <summary>
        /// get handle var.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="dtype"></param>
        /// <param name="scope"></param>
        /// <returns> var. </returns>
        public static Var Handle(string name, DataType dtype, string scope = "") => new Var(name, TensorType.Scalar(new PointerType(dtype)));

        /// <summary>
        /// get the size var. it can be used in tensor shape. like n>=0, m>=0.
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Var SizeVar(string name) => Scalar(name, DataTypes.Int32);
    }
}
