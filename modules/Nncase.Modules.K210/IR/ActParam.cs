namespace Nncase.IR;

public class ActParam
{
    public float[] X0;
    public float[] Kl;
    public float[] Bl;
    public float[] Kr;
    public float[] Br;

    public int Channels => X0.Length;

    public bool BiasOnly => Kl.All(x => x == 1f) && Kr.All(x => x == 1f);

    public ActParam(int c)
    {
        var zeros = Enumerable.Repeat(0, c).Select(x => (float)x);
        var ones = Enumerable.Repeat(1, c).Select(x => (float)x);
        X0 = zeros.ToArray();
        Bl = zeros.ToArray();
        Br = zeros.ToArray();
        Kl = ones.ToArray();
        Kr = ones.ToArray();
    }

    public void ForEachChannel(Action<ActParam, int> f)
    {
        for (int i = 0; i < Channels; i++)
        {
            f(this, i);
        }
    }

    public float[] AsActData()
    {
        var result = new float[Kl.Length * 5];
        for (int i = 0; i < Kl.Length; i++)
        {
            int baseI = i * 5;
            result[baseI + 0] = X0[i];
            result[baseI + 1] = Kl[i];
            result[baseI + 2] = Bl[i];
            result[baseI + 3] = Kr[i];
            result[baseI + 4] = Br[i];
        }

        return result;
    }

    public Tensor ToFakeActData(bool abs = false)
    {
        return Tensor.FromSpan(
          AsActData()
            .Select(x => (abs && System.Math.Abs(x) < 1e-30) ? 0f : x)
            .ToArray(),
          new[] { Channels, 5 });
    }

    public Tensor ToActData(bool abs = false)
    {
        return Tensor.FromSpan(
            AsActData()
                .Select(x => (abs && System.Math.Abs(x) < 1e-30) ? 0f : x)
                .Select(BFloat16.RoundToBFloat16)
                .ToArray(),
            new[] { Channels, 5 });
    }

    public static ActParam FromTensor(Tensor<float> t)
    {
        if (t.Shape[^1] != 5 || t.Shape.Rank != 2)
        {
            throw new InvalidOperationException("InvalidActParamTensor");
        }

        var oc = t.Shape[^2].FixedValue;
        var act = new ActParam(oc);
        act.ForEachChannel((actParam, c) =>
        {
            actParam.X0[c] = t[c, 0];
            actParam.Kl[c] = t[c, 1];
            actParam.Bl[c] = t[c, 2];
            actParam.Kr[c] = t[c, 3];
            actParam.Br[c] = t[c, 4];
        });
        return act;
    }

    public static float[] Add(float[] a, float[] b)
    {
        if (a.Length != b.Length)
        {
            throw new InvalidOperationException("A and B shoule have same Length");
        }
        return a.Zip(b).Select(x => x.Item1 + x.Item2).ToArray();
    }
}