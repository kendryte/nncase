using System;
using System.Collections.Generic;
using System.Text;

namespace NnCase.Targets.K210
{
    public enum KPUFilterType
    {
        Filter_1x1 = 0,
        Filter_3x3 = 1
    }

    public enum KPUPoolType
    {
        Pool_Bypass = 0,
        Pool_Max_2_S2 = 1,
        Pool_Mean_2_S2 = 2,
        Pool_Max_4_S4 = 3,
        Pool_Mean_4_S4 = 4,
        Pool_LeftTop_2_S2 = 5,
        Pool_RightTop_2_S2 = 6,
        Pool_LeftTop_4_S4 = 7,
        Pool_Mean_2_S1 = 8,
        Pool_Max_2_S1 = 9
    }
}
