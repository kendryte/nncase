using NetFabric.Hyperlinq;
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
        return (float) (sum / (v1 * v2));
    }

    public static bool TensorValueCompare(TensorValue pre, TensorValue post, float thresh)
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
        (TensorValue a, TensorValue b) => TensorValueCompare(a, b, thresh),
        (_, _) => throw new ArgumentOutOfRangeException()
    };

    public static float[] CompareByChannel(IValue pre, IValue post, int channelAxis = 1, float thresh = 0.99f) =>
        CompareByChannel(pre.AsTensor(), post.AsTensor(), channelAxis, thresh);

    public static float[] CompareByChannel(Tensor pre, Tensor post, int channelAxis = 1, float thresh = 0.99f)
    {
        var v1 = SliceByChannel(pre, channelAxis);
        var v2 = SliceByChannel(post, channelAxis);
        return v1.Zip(v2).Select(data => CosSimilarity(data.Item1, data.Item2)).ToArray();
    }

    public record Deviation(float v1, float v2, int[] index)
    {
        public float devi => Math.Abs(v1 - v2);
        // todo: v2 is 0.0000
        public float ratio => (v1 == 0 && v2 == 0) || (Math.Abs(v1) < 1e-9 && v2 == 0) ? 1 : Math.Abs(Math.Max(v1, v2) / Math.Min(v1, v2));
    }
    
    public static Deviation[] Deviations(Tensor pre, Tensor post)
    {
        var v1 = pre.ToArray<float>();
        var v2 = post.ToArray<float>();
        return v1.Zip(v2).Select((data, i) => new Deviation(data.Item1, data.Item2, new[] {i})).ToArray();
    }

    private static Tensor SliceTensor(Tensor tensor, int start, int length, int channelAxis = 1)
    {
        var s = tensor.ElementType.SizeInBytes;
        return Tensor.FromBytes(tensor.ElementType,
            tensor.BytesBuffer.Slice(start * s, length * s), tensor.Dimensions[(channelAxis + 1)..]);
    }

    public static Tensor[] SliceByChannel(Tensor tensor, int channelAxis = 1)
    {
        var i = channelAxis + 1;
        var channels = tensor.Dimensions[..i].ToArray().Aggregate(1, (a, b) => a * b);;
        var size = tensor.Dimensions[i..].ToArray().Aggregate(1, (a, b) => a * b);;
        return Enumerable.Range(0, channels).Select(i =>
                SliceTensor(tensor, size * i, size, channelAxis))
            .ToArray();
    }
}

public class LazyCompartor
{
    // count => Either<ErrorReason, CosSim>
    private Dictionary<int, LanguageExt.Either<string, float>> error = new();

    public bool Compare(IValue pre, IValue post, int count, float thresh = 0.99f)
    {
        if (pre is TensorValue a &&
            post is TensorValue b)
        {
            return Compare(a, b, thresh).Match(cos =>
                {
                    error[count] = cos;
                    return false;
                },
                () => true);
        }
        else
        {
            error[count] = $"pre is {pre.GetType().Name}, post is {post.GetType().Name}";
            return false;
        }
    }

    public void AddFailed(int count, string reason)
    {
        error[count] = reason;
    }
    
    public static Option<float> Compare(TensorValue pre, TensorValue post, float thresh)
    {
        var v1 = pre.AsTensor();
        var v2 = post.AsTensor();
        var cosSim = Comparator.CosSimilarity(v1, v2);
        if (cosSim < thresh)
        {
            return Option.Some(cosSim);
        }
        else
        {
            return Option.None;
        }
    }

    public void Run(Action f)
    {
        f();
        FailedAssert();
    }
    
    public void FailedAssert()
    {
        foreach (var (count, value) in error)
        {
            var reason = value.Match(cos => $"CosSim:{cos}", s => s);
            Console.WriteLine($"count:{count} error, reason: ${reason}");
        }

        Assert.True(false);
    }
}