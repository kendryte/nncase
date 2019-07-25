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
            return filterType switch
            {
                KPUFilterType.Filter_1x1 => 1,
                KPUFilterType.Filter_3x3 => 3,
                _ => throw new ArgumentOutOfRangeException(nameof(filterType))
            };
        }

        public static int GetKPUFilterSize(KPUPoolType poolType)
        {
            switch (poolType)
            {
                case KPUPoolType.Pool_Bypass:
                    return 1;
                case KPUPoolType.Pool_Max_2_S2:
                case KPUPoolType.Pool_Mean_2_S2:
                case KPUPoolType.Pool_LeftTop_2_S2:
                case KPUPoolType.Pool_RightTop_2_S2:
                case KPUPoolType.Pool_Mean_2_S1:
                case KPUPoolType.Pool_Max_2_S1:
                    return 2;
                case KPUPoolType.Pool_Max_4_S4:
                case KPUPoolType.Pool_Mean_4_S4:
                case KPUPoolType.Pool_LeftTop_4_S4:
                    return 4;
                default:
                    throw new ArgumentOutOfRangeException(nameof(poolType));
            }
        }

        public static int GetKPUFilterStride(KPUPoolType poolType)
        {
            switch (poolType)
            {
                case KPUPoolType.Pool_Bypass:
                case KPUPoolType.Pool_Mean_2_S1:
                case KPUPoolType.Pool_Max_2_S1:
                    return 1;
                case KPUPoolType.Pool_Max_2_S2:
                case KPUPoolType.Pool_Mean_2_S2:
                case KPUPoolType.Pool_LeftTop_2_S2:
                case KPUPoolType.Pool_RightTop_2_S2:
                    return 2;
                case KPUPoolType.Pool_Max_4_S4:
                case KPUPoolType.Pool_Mean_4_S4:
                case KPUPoolType.Pool_LeftTop_4_S4:
                    return 4;
                default:
                    throw new ArgumentOutOfRangeException(nameof(poolType));
            }
        }

        public static (int h, int w) GetKPUSelectPoolOffset(KPUPoolType poolType)
        {
            switch (poolType)
            {
                case KPUPoolType.Pool_LeftTop_2_S2:
                case KPUPoolType.Pool_LeftTop_4_S4:
                    return (0, 0);
                case KPUPoolType.Pool_RightTop_2_S2:
                    return (0, 1);
                default:
                    throw new ArgumentOutOfRangeException(nameof(poolType));
            }
        }

        public static int GetKPUOutputSize(int input, KPUPoolType poolType)
        {
            return input / GetKPUFilterStride(poolType);
        }

        public static bool IsSupportedShape(Shape shape)
        {
            return shape[1] < 1024 && shape[2] >= 4 && shape[2] <= 256 && shape[3] >= 4 && shape[3] <= 512;
        }
    }
}
