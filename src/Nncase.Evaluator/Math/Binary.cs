// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Binary"/>.
/// </summary>
public partial class BinaryEvaluator : IEvaluator<Binary>, ITypeInferencer<Binary>, ICostEvaluator<Binary>, IOpPrinter<Binary>, IMetricEvaluator<Binary>
{
    public static IRType CheckSBP(BinaryOp op, TensorType tensorType, DistributedType a, DistributedType b)
    {
        // assume broadcast shapes are left algin
        var padA = tensorType.Shape.Rank - a.TensorType.Shape.Rank;
        var padB = tensorType.Shape.Rank - b.TensorType.Shape.Rank;
        var ndsbp = new SBP[tensorType.Shape.Rank];
        for (int i = 0; i < ndsbp.Length; i++)
        {
            var policyA = i < padA ? null : a.AxisPolicies[i - padA];
            var policyB = i < padB ? null : b.AxisPolicies[i - padB];
            switch (policyA, policyB)
            {
                case (null, _):
                    ndsbp[i] = policyB!;
                    break;
                case (_, null):
                    ndsbp[i] = policyA!;
                    break;
                case (SBPSplit sa, SBPSplit sb):
                    if (sa.Axes != sb.Axes)
                    {
                        return new InvalidType($"lhs rhs sbp at {i} not equal");
                    }

                    ndsbp[i] = sa;
                    break;
                case (SBPSplit sa, SBPBroadCast):
                    // invalid (S, B) if B is not broacast
                    if (b.TensorType.Shape[i - padB] is { IsFixed: false } || (b.TensorType.Shape[i - padB] is { IsFixed: true, FixedValue: var fb } && fb != 1))
                    {
                        return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                    }

                    ndsbp[i] = sa;
                    break;
                case (SBPBroadCast, SBPSplit sb):
                    // invalid (B, S) if A is not broacast
                    if (a.TensorType.Shape[i - padA] is { IsFixed: false } || (a.TensorType.Shape[i - padA] is { IsFixed: true, FixedValue: var fa } && fa != 1))
                    {
                        return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                    }

                    ndsbp[i] = sb;
                    break;
                case (SBPBroadCast, SBPBroadCast):
                    ndsbp[i] = SBP.B;
                    break;
                default:
                    return new InvalidType("not support binary sbp.");
            }
        }

        if (!DistributedUtility.IsDistributable(ndsbp))
        {
            return new InvalidType("not support binary sbp.");
        }

        return new DistributedType(tensorType, ndsbp, a.Placement);
    }

    /// <inheritdoc />
    public IValue Visit(IEvaluateContext context, Binary binary)
    {
        var lhs = context.GetArgumentValueAsTensor(binary, Binary.Lhs);
        var rhs = context.GetArgumentValueAsTensor(binary, Binary.Rhs);
        var originDtype = lhs.ElementType;

        if (lhs.Shape.IsScalar && rhs.Shape.IsScalar)
        {
            if (lhs.ElementType == DataTypes.Int32 && rhs.ElementType == DataTypes.Int32)
            {
                return Value.FromTensor(Tensor.FromScalar(Compute(binary.BinaryOp, lhs.ToScalar<int>(), rhs.ToScalar<int>())));
            }
            else if (lhs.ElementType == DataTypes.Int64 && rhs.ElementType == DataTypes.Int64)
            {
                return Value.FromTensor(Tensor.FromScalar(Compute(binary.BinaryOp, lhs.ToScalar<long>(), rhs.ToScalar<long>())));
            }
            else if (lhs.ElementType == DataTypes.Float32 && rhs.ElementType == DataTypes.Float32)
            {
                return Value.FromTensor(Tensor.FromScalar(Compute(binary.BinaryOp, lhs.ToScalar<float>(), rhs.ToScalar<float>())));
            }
            else if (lhs.ElementType == DataTypes.Boolean && rhs.ElementType == DataTypes.Boolean)
            {
                return Value.FromTensor(Tensor.FromScalar(Compute(binary.BinaryOp, lhs.ToScalar<bool>(), rhs.ToScalar<bool>())));
            }
            else if (lhs.ElementType == DataTypes.UInt32 && rhs.ElementType == DataTypes.UInt32)
            {
                return Value.FromTensor(Tensor.FromScalar(Compute(binary.BinaryOp, lhs.ToScalar<uint>(), rhs.ToScalar<uint>())));
            }
            else if (lhs.ElementType is PointerType && (rhs.ElementType == DataTypes.UInt32 || rhs.ElementType == DataTypes.UInt64))
            {
                return Value.FromTensor(Tensor.FromScalar(Compute(binary.BinaryOp, lhs.ToScalar<ulong>(), rhs.ToScalar<ulong>())));
            }
            else if ((lhs.ElementType == DataTypes.UInt32 || lhs.ElementType == DataTypes.UInt64) && rhs.ElementType is PointerType)
            {
                return Value.FromTensor(Tensor.FromScalar(Compute(binary.BinaryOp, lhs.ToScalar<ulong>(), rhs.ToScalar<ulong>())));
            }
            else
            {
                return Ort_compute(binary, lhs, rhs, originDtype);
            }
        }

        // for float16/float8/bfloat16 infere
        if (originDtype.IsFloat())
        {
            lhs = lhs.CastElement<float>();
            rhs = rhs.CastElement<float>();
        }

        return Ort_compute(binary, lhs, rhs, originDtype);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Binary target)
    {
        var lhs = context.CheckArgumentType<IRType>(target, Binary.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, Binary.Rhs);
        return (lhs, rhs) switch
        {
            (TensorType a, TensorType b) => Visit(target, a, b),
            (DistributedType a, DistributedType b) => Visit(target, a, b),
            (AnyType, _) => AnyType.Default,
            (_, AnyType) => AnyType.Default,
            _ => new InvalidType($"{lhs} {rhs}"),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Binary target)
    {
        var lhsType = context.GetArgumentType<IRType>(target, Binary.Lhs);
        var rhsType = context.GetArgumentType<IRType>(target, Binary.Rhs);
        var outputType = context.GetReturnType<IRType>();

        return new()
        {
            [CostFactorNames.MemoryLoad] = CostUtility.GetMemoryAccess(lhsType) + CostUtility.GetMemoryAccess(rhsType),
            [CostFactorNames.MemoryStore] = CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(outputType, CostUtility.GetCPUCyclesOfBinary(target.BinaryOp)),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Binary target)
    {
        var lhsType = context.GetArgumentType<TensorType>(target, Binary.Lhs);
        var rhsType = context.GetArgumentType<TensorType>(target, Binary.Rhs);
        var outputType = context.GetReturnType<TensorType>();

        return new()
        {
            [MetricFactorNames.OffChipMemoryTraffic] = CostUtility.GetMemoryAccess(lhsType) + CostUtility.GetMemoryAccess(rhsType) + CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(outputType, (int)MetricUtility.GetBinaryFLOPs(target.BinaryOp)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, Binary target)
    {
        var lhs = context.GetArgument(target, Binary.Lhs);
        var rhs = context.GetArgument(target, Binary.Rhs);
        if (context.Flags.HasFlag(PrinterFlags.Inline) || context.Flags.HasFlag(PrinterFlags.Script))
        {
            return target.BinaryOp switch
            {
                BinaryOp.Add => $"({lhs} + {rhs})",
                BinaryOp.Sub => $"({lhs} - {rhs})",
                BinaryOp.Mul => $"({lhs} * {rhs})",
                BinaryOp.Div => $"({lhs} / {rhs})",
                BinaryOp.Mod => $"({lhs} % {rhs})",
                BinaryOp.LogicalAnd => $"({lhs} & {rhs})",
                BinaryOp.LogicalOr => $"({lhs} | {rhs})",
                BinaryOp.LogicalXor => $"({lhs} ^ {rhs})",
                BinaryOp.LeftShift => $"({lhs} << {rhs})",
                BinaryOp.RightShift => $"({lhs} >> {rhs})",
                _ => $"{target.BinaryOp}({lhs}, {rhs})",
            };
        }

        return $"{target.BinaryOp}({lhs}, {rhs})";
    }

    private IRType Visit(Binary target, DistributedType a, DistributedType b)
    {
        if (a.Placement != b.Placement)
        {
            return new InvalidType("lhs rhs have different placement");
        }

        var rType = Visit(target, a.TensorType, b.TensorType);
        if (rType is not TensorType tensorType)
        {
            return rType;
        }

        return CheckSBP(target.BinaryOp, tensorType, a, b);
    }

    private int Compute(BinaryOp op, int a, int b) => op switch
    {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => (int)System.Math.Floor((float)a / b),
        BinaryOp.CeilDiv => (int)System.Math.Ceiling((float)a / b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => checked((int)System.Math.Pow(a, b)),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private uint Compute(BinaryOp op, uint a, uint b) => op switch
    {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => (uint)System.Math.Floor((float)a / b),
        BinaryOp.CeilDiv => (uint)System.Math.Ceiling((float)a / b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => checked((uint)System.Math.Pow((double)a, (double)b)),
        BinaryOp.LeftShift => a << (int)b,
        BinaryOp.RightShift => a >> (int)b,
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private ulong Compute(BinaryOp op, ulong a, ulong b) => op switch
    {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => (ulong)System.Math.Floor((float)a / b),
        BinaryOp.CeilDiv => (ulong)System.Math.Ceiling((float)a / b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private bool Compute(BinaryOp op, bool a, bool b) => op switch
    {
        BinaryOp.LogicalAnd => a & b,
        BinaryOp.LogicalOr => a | b,
        BinaryOp.LogicalXor => a ^ b,
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private long Compute(BinaryOp op, long a, long b) => op switch
    {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => (long)System.Math.Floor((float)a / b),
        BinaryOp.CeilDiv => (long)System.Math.Ceiling((float)a / b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => checked((int)System.Math.Pow(a, b)),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private float Compute(BinaryOp op, float a, float b) => op switch
    {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => System.MathF.Pow(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private IValue Ort_compute(Binary binary, Tensor lhs, Tensor rhs, DataType dataType)
    {
        var a = lhs.ToOrtTensor();
        var b = rhs.ToOrtTensor();
        if (lhs.ElementType is VectorType vt && rhs.ElementType is not VectorType)
        {
            b.Reshape(b.Shape.Concat(Enumerable.Repeat(1L, vt.Lanes.Count)).ToArray());
        }

        if (rhs.ElementType is VectorType vt2 && lhs.ElementType is not VectorType)
        {
            a.Reshape(a.Shape.Concat(Enumerable.Repeat(1L, vt2.Lanes.Count)).ToArray());
        }

        static OrtKISharp.Tensor Mod(OrtKISharp.Tensor a, OrtKISharp.Tensor b)
        {
            var fmod = DataTypes.IsFloat(a.DataType.ToDataType()) && DataTypes.IsFloat(b.DataType.ToDataType()) ? 1L : 0L;
            return OrtKI.Mod(a, b, fmod);
        }

        return (binary.BinaryOp switch
        {
            BinaryOp.Add => a + b,
            BinaryOp.Sub => a - b,
            BinaryOp.Mul => a * b,
            BinaryOp.Div => a / b,
            BinaryOp.FloorDiv => OrtKI.Floor(a.Cast(OrtDataType.Float) / b.Cast(OrtDataType.Float)).Cast(a.DataType),
            BinaryOp.CeilDiv => OrtKI.Ceil(a.Cast(OrtDataType.Float) / b.Cast(OrtDataType.Float)).Cast(a.DataType),
            BinaryOp.Mod => Mod(a, b),
            BinaryOp.Min => OrtKI.Min(new[] { a, b }),
            BinaryOp.Max => OrtKI.Max(new[] { a, b }),
            BinaryOp.Pow => OrtKI.Pow(a, b),
            BinaryOp.BitwiseAnd => throw new NotSupportedException("NotSupported BinaryOp BitwiseAnd"),
            BinaryOp.BitwiseOr => throw new NotSupportedException("NotSupported BinaryOp BitwiseOr"),
            BinaryOp.BitwiseXor => throw new NotSupportedException("NotSupported BinaryOp BitwiseXor"),
            BinaryOp.LogicalAnd => OrtKI.And(a, b),
            BinaryOp.LogicalOr => OrtKI.Or(a, b),
            BinaryOp.LogicalXor => OrtKI.Xor(a, b),
            BinaryOp.LeftShift => OrtKI.LeftShift(a, b),
            BinaryOp.RightShift => OrtKI.RightShift(a, b),
            _ => throw new ArgumentOutOfRangeException(nameof(binary)),
        }).ToValue(dataType);
    }

    private IRType Visit(Binary target, TensorType lhs, TensorType rhs)
    {
        if (target.BinaryOp is BinaryOp.LeftShift or BinaryOp.RightShift && (lhs.DType != DataTypes.UInt32 || rhs.DType != DataTypes.UInt32))
        {
            return new InvalidType("The Binary LeftShift RightShift Only Accept The UInt32 Datatype.");
        }

        if ((target.BinaryOp is BinaryOp.LogicalAnd or BinaryOp.LogicalOr or BinaryOp.LogicalXor) &&
            (lhs.DType != DataTypes.Boolean || rhs.DType != DataTypes.Boolean))
        {
            return new InvalidType("The Binary Logical Only Accept The Boolean Datatype.");
        }

        if (lhs is { DType: PointerType { ElemType: var letype } })
        {
            if ((rhs is { DType: PointerType { ElemType: var other } } && letype == other) || rhs.DType == DataTypes.UInt64 || rhs.DType == DataTypes.UInt32)
            {
                return TensorType.Pointer(letype);
            }

            return new InvalidType($"The Binary Lhs {CompilerServices.Print(lhs)} != Rhs {CompilerServices.Print(rhs)}");
        }

        if (rhs is { DType: PointerType { ElemType: var retype } })
        {
            if ((lhs is { DType: PointerType { ElemType: var other } } && retype == other) || lhs.DType == DataTypes.UInt64 || lhs.DType == DataTypes.UInt32)
            {
                return TensorType.Pointer(retype);
            }

            return new InvalidType($"The Binary Lhs {CompilerServices.Print(lhs)} != Rhs {CompilerServices.Print(rhs)}");
        }

        return TypeInference.BroadcastType(lhs, rhs);
    }
}
