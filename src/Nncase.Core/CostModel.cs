using System;
using Nncase.IR;

namespace Nncase
{
    public class CostModelContext
    {
        public Const GetArgument(Op op, ParameterInfo info)
        {
            throw new NotImplementedException();
        }
    }
    
    public sealed record CostModel
    {
        public CostModel(long macCount, long dataMoveCount = 0)
        {
            MacCount = macCount;
            DataMoveCount = dataMoveCount;
        }

        public long MacCount
        {
            get => MacCount;
            private set => MacCount = value;
        }
        
        public long DataMoveCount
        {
            get => DataMoveCount;
            private set => DataMoveCount = value;
        }
    }
}