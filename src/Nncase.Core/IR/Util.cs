namespace Nncase.IR
{
    public class Util
    {
        public static int PositiveIndex(int index, TensorType input)
        {
            return PositiveIndex(index, input.Shape.Rank);
        }
        
        public static int PositiveIndex(int index, int rank)
        {
            return index < 0 ? index + rank : index;
        }
    }
}