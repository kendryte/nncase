namespace Nncase.IR.Tensors
{
    public record Paddings()
    {
        public int before = 0;
        public int after = 0;

        public int Sum => before + after;

        public static (Paddings, Paddings) GetPaddingFromConst(Const c)
        {
            var v = c.ToTensor<int>();
            var padH = new Paddings();
            var padW = new Paddings();
            padH.before = v[0, 0];
            padH.after = v[0, 1];
            padW.before = v[1, 0];
            padW.after = v[1, 1];
            return (padH, padW);
        }
    }
}