using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct image_size_t
    {
        public ulong Value;

        private const int i_row_widShift = 0;
        private const ulong i_row_widMask = unchecked((ulong)((1UL << 10) - (1UL << 0)));
        public Bit10 i_row_wid
        {
            get => (Bit10)((Value & i_row_widMask) >> i_row_widShift);
            set => Value = unchecked((ulong)((Value & ~i_row_widMask) | ((((ulong)value) << i_row_widShift) & i_row_widMask)));
        }
        private const int i_col_highShift = 10;
        private const ulong i_col_highMask = unchecked((ulong)((1UL << 19) - (1UL << 10)));
        public Bit9 i_col_high
        {
            get => (Bit9)((Value & i_col_highMask) >> i_col_highShift);
            set => Value = unchecked((ulong)((Value & ~i_col_highMask) | ((((ulong)value) << i_col_highShift) & i_col_highMask)));
        }
        private const int reserved0Shift = 19;
        private const ulong reserved0Mask = unchecked((ulong)((1UL << 32) - (1UL << 19)));
        public Bit13 reserved0
        {
            get => (Bit13)((Value & reserved0Mask) >> reserved0Shift);
            set => Value = unchecked((ulong)((Value & ~reserved0Mask) | ((((ulong)value) << reserved0Shift) & reserved0Mask)));
        }
        private const int o_row_widShift = 32;
        private const ulong o_row_widMask = unchecked((ulong)((1UL << 42) - (1UL << 32)));
        public Bit10 o_row_wid
        {
            get => (Bit10)((Value & o_row_widMask) >> o_row_widShift);
            set => Value = unchecked((ulong)((Value & ~o_row_widMask) | ((((ulong)value) << o_row_widShift) & o_row_widMask)));
        }
        private const int o_col_highShift = 42;
        private const ulong o_col_highMask = unchecked((ulong)((1UL << 51) - (1UL << 42)));
        public Bit9 o_col_high
        {
            get => (Bit9)((Value & o_col_highMask) >> o_col_highShift);
            set => Value = unchecked((ulong)((Value & ~o_col_highMask) | ((((ulong)value) << o_col_highShift) & o_col_highMask)));
        }
        private const int reserved1Shift = 51;
        private const ulong reserved1Mask = unchecked((ulong)((1UL << 64) - (1UL << 51)));
        public Bit13 reserved1
        {
            get => (Bit13)((Value & reserved1Mask) >> reserved1Shift);
            set => Value = unchecked((ulong)((Value & ~reserved1Mask) | ((((ulong)value) << reserved1Shift) & reserved1Mask)));
        }
    }
}
