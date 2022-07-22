using Xunit;

namespace Nncase.TestFixture;

public static class Comparator
{
    private static float Prod(float[] data1, float[] data2)
    {
        return data1.Zip(data2).Aggregate(0f, (f, tuple) => f + tuple.Item1 * tuple.Item2);
    }

    public static float CosSimilarity(IValue a, IValue b)
    {
        return CosSimilarity(a.AsTensor(), b.AsTensor());
    }

    public static float CosSimilarity(Tensor a, Tensor b)
    {
        var va = a.ToArray<float>();
        var vb = b.ToArray<float>();
        var v1 = Math.Sqrt(Prod(va, va));
        var v2 = Math.Sqrt(Prod(vb, vb));
        var sum = Prod(va, vb);
        return (float)(sum / (v1 * v2));
    }

    public static bool Compare(TensorValue pre, TensorValue post, float thresh)
    {
        var v1 = pre.AsTensor();
        var v2 = post.AsTensor();
        Assert.Equal(v1.Shape, v2.Shape);
        Assert.Equal(v1.ElementType, v2.ElementType);
        var cosSim = CosSimilarity(v1, v2);
        Assert.True(cosSim > thresh, "cosSim:" + cosSim);
        return true;
    }

    public static bool Compare(IValue pre, IValue post, float thresh = 0.99f) => (pre, post) switch
    {
        (TensorValue a, TensorValue b) => Compare(a, b, thresh),
        (_, _) => throw new ArgumentOutOfRangeException()
    };
}