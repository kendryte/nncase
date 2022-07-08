using Nncase.IR;
using Random = Nncase.IR.F.Random;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.NN;
namespace Nncase.TestFixture;

public static class DataGenerator
{
    private static System.Random rand = new();

    public static IEnumerable<T> EnumValues<T>()
    {
        return Enum.GetValues(typeof(T)).Cast<T>();
    }
    
    public static Expr DefaultRandom()
    {
        return DefaultRandom(DataTypes.Float32, new[] { 3, DefaultChannel, 4, 8 });
    }

    public static int DefaultChannel => 2;
    public static int[] DefaultShape => new[] { 3, 2, 4, 8 };

    public static Expr DefaultRandom(DataType dt)
    {
        return DefaultRandom(dt, DefaultShape);
    }

    public static Expr DefaultRandom(int[] shape)
    {
        return DefaultRandom(DataTypes.Float32, shape);
    }

    public static Expr DefaultRandom(DataType dt, int[] shape)
    {
        // if (dt.IsIntegral())
        // {
        //     return Testing.Rand(dt, shape);
        // }

        return
            Random.Normal(DataTypes.Float32, new Shape(shape)).Evaluate().AsTensor().CastTo(dt);
    }

    public static Expr RandomLimitOne()
    {
        return Sigmoid(Random.Normal(DataTypes.Float32, new Shape(DefaultShape)));
    }

    public static Expr RandomGNNEScalar()
    {
        return Sigmoid(
            Random.Normal(DataTypes.Float32, new Shape(1, 1, 1, 1))).Evaluate().AsTensor();
    }

    public static Expr RandomScalar()
    {
        return Sigmoid(
            Random.Normal(DataTypes.Float32, new[] { 1 })).Evaluate().AsTensor();
    }

    // nncase format DeQuantizeParam
    public static QuantParam RandomQuantParam()
    {
        var qp = new QuantParam(rand.Next(1, 2), rand.Next(55, 255));
        return qp with { Scale = 1 / qp.Scale };
    }

    public static Expr DefaultConv()
    {
        var input = Random.Normal(DataTypes.Float32, new[] { 1, 3, 24, 32 });
        var weights = Random.Normal(DataTypes.Float32, new[] { 16, 3, 3, 3 }).Evaluate();
        var bias = Random.Normal(DataTypes.Float32, new[] { 16 }).Evaluate();
        var stride = Tensor.FromSpan(new[] { 1, 1 }, new[] { 2 });
        var dilation = Tensor.FromSpan(new[] { 1, 1 }, new[] { 2 });
        var padding = new[,] { { 0, 1 }, { 0, 0 } };

        var conv = Conv2D(input, weights.AsTensor(), bias.AsTensor(), stride, padding,
            dilation,
            PadMode.Constant, 1);
        return conv;
    }
    
     public static IEnumerable<object[]> ResizeModeProduct()
     {
         var v1 = EnumValues<ImageResizeMode>().Select(x => (object) x).ToArray();
         var v2 = EnumValues<ImageResizeNearestMode>().Select(x => (object) x).ToArray();
         var v3 = EnumValues<ImageResizeTransformationMode>().Select(x => (object) x).ToArray();
         var result = Product(
             new[] {v1, v2, v3}).Select(x => (object[]) x.ToArray());
         return result;
     }
     
     public static IEnumerable<IEnumerable<T>> Product<T>
         (this IEnumerable<IEnumerable<T>> sequences)
     {
         IEnumerable<IEnumerable<T>> emptyProduct =
             new[] { Enumerable.Empty<T>() };
         var ret = sequences.Aggregate(
             emptyProduct,
             (accumulator, sequence) =>
                 from accseq in accumulator
                 from item in sequence
                 select accseq.Concat(new[] { item }));
         return ret;
     }
}