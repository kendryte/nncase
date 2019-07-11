using System;
using System.Collections.Generic;
using System.Text;
using NnCase.IR;

namespace NnCase.Targets.K210.IR
{
    public static class KPUShapeUtility
    {
        public static int GetKPUPadding(KPUFilterType filterType)
        {
            switch (filterType)
            {
                case KPUFilterType.Filter_1x1:
                    return 0;
                case KPUFilterType.Filter_3x3:
                    return 1;
                default:
                    throw new ArgumentOutOfRangeException(nameof(filterType));
            }
        }

        public static int GetKPUFilterSize(KPUFilterType filterType)
        {
            switch (filterType)
            {
                case KPUFilterType.Filter_1x1:
                    return 1;
                case KPUFilterType.Filter_3x3:
                    return 3;
                default:
                    throw new ArgumentOutOfRangeException(nameof(filterType));
            }
        }

        public static int GetKPUOutputSize(int input, KPUPoolType poolType)
        {
            int stride;
            switch (poolType)
            {
                case KPUPoolType.Pool_Bypass:
                case KPUPoolType.Pool_Mean_2_S1:
                case KPUPoolType.Pool_Max_2_S1:
                    stride = 1;
                    break;
                case KPUPoolType.Pool_Max_2_S2:
                case KPUPoolType.Pool_Mean_2_S2:
                case KPUPoolType.Pool_LeftTop_2_S2:
                case KPUPoolType.Pool_RightTop_2_S2:
                    stride = 2;
                    break;
                case KPUPoolType.Pool_Max_4_S4:
                case KPUPoolType.Pool_Mean_4_S4:
                case KPUPoolType.Pool_LeftTop_4_S4:
                    stride = 4;
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(poolType));
            }

            return input / stride;
        }

        public static bool IsSupportedShape(Shape shape)
        {
            return shape[1] < 1024 && shape[2] <= 512 && shape[3] <= 256;
        }
    }
}
