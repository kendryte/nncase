using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase
{
    /// <summary>
    /// Binary opeartor kind.
    /// </summary>
    public enum BinaryOp
    {
        /// <summary>
        /// Add.
        /// </summary>
        Add,

        /// <summary>
        /// Sub.
        /// </summary>
        Sub,

        /// <summary>
        /// Multiply.
        /// </summary>
        Mul,

        /// <summary>
        /// Divide.
        /// </summary>
        Div,

        /// <summary>
        /// Modulus.
        /// </summary>
        Mod,

        /// <summary>
        /// Minimum.
        /// </summary>
        Min,

        /// <summary>
        /// Maximum.
        /// </summary>
        Max,

        /// <summary>
        /// Power.
        /// </summary>
        Pow,

        /// <summary>
        /// Bitwise and.
        /// </summary>
        BitwiseAnd,

        /// <summary>
        /// Bitwise or.
        /// </summary>
        BitwiseOr,

        /// <summary>
        /// Bitwise xor.
        /// </summary>
        BitwiseXor,

        /// <summary>
        /// Logical and.
        /// </summary>
        LogicalAnd,

        /// <summary>
        /// Logical or.
        /// </summary>
        LogicalOr,

        /// <summary>
        /// Logical xor.
        /// </summary>
        LogicalXor,
    }
}
