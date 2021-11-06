using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Transform.Pattern.Math;
using Nncase.Transform.Pattern.NN;
using Nncase.Transform.Pattern.Tensors;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;

namespace Nncase.Transform.Pattern.F
{
    public static partial class Math
    {
        /// <summary>
        /// CallPattern unary.
        /// </summary>
        /// <param name = "unaryOp">Unary operator.</param>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Unary(UnaryOp unaryOp, ExprPattern expr)
        {
            return new UnaryWrapper(new CallPattern(new UnaryPattern(unaryOp), expr));
        }

        /// <summary>
        /// CallPattern binary.
        /// </summary>
        /// <param name = "binaryOp">Binary operator.</param>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Binary(BinaryOp binaryOp, ExprPattern lhs, ExprPattern rhs)
        {
            return new BinaryWrapper(new CallPattern(new BinaryPattern(binaryOp), lhs, rhs));
        }

        /// <summary>
        /// CallPattern clamp.
        /// </summary>
        /// <param name = "input">Input expression.</param>
        /// <param name = "min">Left operand.</param>
        /// <param name = "max">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static ClampWrapper Clamp(ExprPattern input, ExprPattern min, ExprPattern max)
        {
            return new ClampWrapper(new CallPattern(new ClampPattern(), input, min, max));
        }

        /// <summary>
        /// CallPattern clamp.
        /// </summary>
        /// <param name = "input">Input expression.</param>
        /// <param name = "range">Value range.</param>
        /// <typeparam name = "T">Data type.</typeparam>
        /// <returns>Result expression.</returns>
        public static ClampWrapper Clamp<T>(ExprPattern input, ValueRange<T> range)
            where T : unmanaged
        {
            return new ClampWrapper(new CallPattern(new ClampPattern(), input, Const.FromScalar(range.Min), Const.FromScalar(range.Max)));
        }

        /// <summary>
        /// CallPattern abs.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Abs(ExprPattern expr) => Unary(UnaryOp.Abs, expr);
        /// <summary>
        /// CallPattern ceil.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Ceil(ExprPattern expr) => Unary(UnaryOp.Ceil, expr);
        /// <summary>
        /// CallPattern cos.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Cos(ExprPattern expr) => Unary(UnaryOp.Cos, expr);
        /// <summary>
        /// CallPattern exp.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Exp(ExprPattern expr) => Unary(UnaryOp.Exp, expr);
        /// <summary>
        /// CallPattern floor.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Floor(ExprPattern expr) => Unary(UnaryOp.Floor, expr);
        /// <summary>
        /// CallPattern log.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Log(ExprPattern expr) => Unary(UnaryOp.Log, expr);
        /// <summary>
        /// CallPattern neg.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Neg(ExprPattern expr) => Unary(UnaryOp.Neg, expr);
        /// <summary>
        /// CallPattern round.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Round(ExprPattern expr) => Unary(UnaryOp.Round, expr);
        /// <summary>
        /// CallPattern rsqrt.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Rsqrt(ExprPattern expr) => Unary(UnaryOp.Rsqrt, expr);
        /// <summary>
        /// CallPattern sin.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Sin(ExprPattern expr) => Unary(UnaryOp.Sin, expr);
        /// <summary>
        /// CallPattern sqrt.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Sqrt(ExprPattern expr) => Unary(UnaryOp.Sqrt, expr);
        /// <summary>
        /// CallPattern square.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Square(ExprPattern expr) => Unary(UnaryOp.Square, expr);
        /// <summary>
        /// CallPattern tanh.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper Tanh(ExprPattern expr) => Unary(UnaryOp.Tanh, expr);
        /// <summary>
        /// CallPattern bitwise not.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper BitwiseNot(ExprPattern expr) => Unary(UnaryOp.BitwiseNot, expr);
        /// <summary>
        /// CallPattern logical not.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper LogicalNot(ExprPattern expr) => Unary(UnaryOp.LogicalNot, expr);
        /// <summary>
        /// CallPattern add.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Add(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Add, lhs, rhs);
        /// <summary>
        /// CallPattern sub.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Sub(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Sub, lhs, rhs);
        /// <summary>
        /// CallPattern mul.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Mul(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Mul, lhs, rhs);
        /// <summary>
        /// CallPattern div.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Div(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Div, lhs, rhs);
        /// <summary>
        /// CallPattern mod.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Mod(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Mod, lhs, rhs);
        /// <summary>
        /// CallPattern min.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Min(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Min, lhs, rhs);
        /// <summary>
        /// CallPattern max.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Max(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Max, lhs, rhs);
        /// <summary>
        /// CallPattern pow.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper Pow(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.Pow, lhs, rhs);
        /// <summary>
        /// CallPattern bitwise and.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper BitwiseAnd(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseAnd, lhs, rhs);
        /// <summary>
        /// CallPattern bitwise or.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper BitwiseOr(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseOr, lhs, rhs);
        /// <summary>
        /// CallPattern bitwise xor.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper BitwiseXor(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.BitwiseXor, lhs, rhs);
        /// <summary>
        /// CallPattern logical and.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper LogicalAnd(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalAnd, lhs, rhs);
        /// <summary>
        /// CallPattern logical or.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper LogicalOr(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalOr, lhs, rhs);
        /// <summary>
        /// CallPattern logical xor.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper LogicalXor(ExprPattern lhs, ExprPattern rhs) => Binary(BinaryOp.LogicalXor, lhs, rhs);
        /// <summary>
        /// CallPattern floor div.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static UnaryWrapper FloorDiv(ExprPattern lhs, ExprPattern rhs) => Floor(lhs / rhs);
        /// <summary>
        /// CallPattern floor mod.
        /// </summary>
        /// <param name = "lhs">Left operand.</param>
        /// <param name = "rhs">Right operand.</param>
        /// <returns>Result expression.</returns>
        public static BinaryWrapper FloorMod(ExprPattern lhs, ExprPattern rhs) => Sub(lhs, (FloorDiv(lhs, rhs) * rhs));
    }

    public static partial class NN
    {
        /// <summary>
        /// CallPattern sigmoid.
        /// </summary>
        /// <param name = "expr">Source expression.</param>
        /// <returns>Result expression.</returns>
        public static SigmoidWrapper Sigmoid(ExprPattern expr) => new SigmoidWrapper(new CallPattern(new SigmoidPattern(), expr));
        public static Conv2DWrapper Conv2D(ExprPattern input, ExprPattern weights, ExprPattern bias, ExprPattern padding, ExprPattern stride, ExprPattern dilation, PadMode padMode) => new Conv2DWrapper(new CallPattern(new Conv2DPattern(padMode), input, padding, stride, dilation));
    }

    public static partial class Tensors
    {
        public static TransposeWrapper Transpose(ExprPattern input, ExprPattern perm) => new TransposeWrapper(new CallPattern(new TransposePattern(), input, perm));
        public static CastWrapper Cast(ExprPattern input, DataType newType) => new CastWrapper(new CallPattern(new CastPattern(newType), input));
        public static ConcatWrapper Concat(TuplePattern input, ExprPattern axis) => new ConcatWrapper(new CallPattern(new ConcatPattern(), input, axis));
        public static GatherWrapper Gather(ExprPattern input, ExprPattern axis, ExprPattern index) => new GatherWrapper(new CallPattern(new GatherPattern(), input, axis, index));
        public static GatherNDWrapper GatherND(ExprPattern input, ExprPattern axis, ExprPattern batch_dims, ExprPattern index) => new GatherNDWrapper(new CallPattern(new GatherNDPattern(), input, axis, batch_dims, index));
        public static MatMulWrapper MatMul(ExprPattern input, ExprPattern other) => new MatMulWrapper(new CallPattern(new MatMulPattern(), input, other));
        /// Pads is Const tensor, shape = [channels, 2(before, after)]
        public static PadWrapper Pad(ExprPattern input, ExprPattern pads, PadMode mode, ExprPattern value) => new PadWrapper(new CallPattern(new PadPattern(mode), input, pads, value));
        public static ReduceWrapper Reduce(ReduceOp reduceOp, ExprPattern input, ExprPattern axis, ExprPattern initValue, ExprPattern keepDims) => new ReduceWrapper(new CallPattern(new ReducePattern(reduceOp), input, axis, initValue, keepDims));
        public static ReduceWrapper ReduceMean(ExprPattern input, ExprPattern axis, ExprPattern initValue, ExprPattern keepDims) => Reduce(ReduceOp.Mean, input, axis, initValue, keepDims);
        public static ReduceWrapper ReduceMin(ExprPattern input, ExprPattern axis, ExprPattern initValue, ExprPattern keepDims) => Reduce(ReduceOp.Min, input, axis, initValue, keepDims);
        public static ReduceWrapper ReduceMax(ExprPattern input, ExprPattern axis, ExprPattern initValue, ExprPattern keepDims) => Reduce(ReduceOp.Min, input, axis, initValue, keepDims);
        public static ReduceWrapper ReduceSum(ExprPattern input, ExprPattern axis, ExprPattern initValue, ExprPattern keepDims) => Reduce(ReduceOp.Sum, input, axis, initValue, keepDims);
        public static ReshapeWrapper Reshape(ExprPattern input, ExprPattern shape) => new ReshapeWrapper(new CallPattern(new ReshapePattern(), input, shape));
        ///https://github.com/onnx/onnx/blob/master/docs/Operators.md#slice
        public static SliceWrapper Slice(ExprPattern input, ExprPattern begins, ExprPattern ends, ExprPattern axes, ExprPattern strides) => new SliceWrapper(new CallPattern(new SlicePattern(), input, begins, ends, axes, strides));
        public static SliceWrapper Slice(ExprPattern input, Const begins, Const ends)
        {
            var axes = Const.FromSpan<int>(Enumerable.Range(0, ends.Rank).ToArray());
            var strides = axes with {Data = new IRBytes(DataTypes.GetBytes<int>(Enumerable.Repeat(1, ends.Rank).ToArray()))};
            return new SliceWrapper(new CallPattern(new SlicePattern(), input, begins, ends, axes, strides));
        }

        /// squeeze input by give dims
        public static SqueezeWrapper Squeeze(ExprPattern input, ExprPattern dims) => new SqueezeWrapper(new CallPattern(new SqueezePattern(), input, dims));
        public static QuantizeWrapper Quantize(ExprPattern input, ExprPattern quantParam, DataType targetType) => new QuantizeWrapper(new CallPattern(new QuantizePattern(targetType), input, quantParam));
        public static DeQuantizeWrapper DeQuantize(ExprPattern input, ExprPattern quantParam, DataType targetType) => new DeQuantizeWrapper(new CallPattern(new DeQuantizePattern(targetType), input, quantParam));
        // same like tensorflow
        public static SpaceToBatchWrapper SpaceToBatch(ExprPattern input, ExprPattern blockShape, ExprPattern paddings) => new SpaceToBatchWrapper(new CallPattern(new SpaceToBatchPattern(), input, blockShape, paddings));
        public static BatchToSpaceWrapper BatchToSpace(ExprPattern input, ExprPattern blockShape, ExprPattern crops) => new BatchToSpaceWrapper(new CallPattern(new BatchToSpacePattern(), input, blockShape, crops));
    }
}