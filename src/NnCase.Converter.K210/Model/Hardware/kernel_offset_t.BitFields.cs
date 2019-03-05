using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct kernel_offset_t
    {
        public ulong Value;

        private const int coef_column_offsetShift = 0;
        private const ulong coef_column_offsetMask = unchecked((ulong)((1UL << 4) - (1UL << 0)));
        public Bit4 coef_column_offset
        {
            get => (Bit4)((Value & coef_column_offsetMask) >> coef_column_offsetShift);
            set => Value = unchecked((ulong)((Value & ~coef_column_offsetMask) | ((((ulong)value) << coef_column_offsetShift) & coef_column_offsetMask)));
        }
        private const int coef_row_offsetShift = 4;
        private const ulong coef_row_offsetMask = unchecked((ulong)((1UL << 16) - (1UL << 4)));
        public Bit12 coef_row_offset
        {
            get => (Bit12)((Value & coef_row_offsetMask) >> coef_row_offsetShift);
            set => Value = unchecked((ulong)((Value & ~coef_row_offsetMask) | ((((ulong)value) << coef_row_offsetShift) & coef_row_offsetMask)));
        }
        private const int reserved0Shift = 16;
        private const ulong reserved0Mask = unchecked((ulong)((1UL << 56) - (1UL << 16)));
        public Bit40 reserved0
        {
            get => (Bit40)((Value & reserved0Mask) >> reserved0Shift);
            set => Value = unchecked((ulong)((Value & ~reserved0Mask) | ((((ulong)value) << reserved0Shift) & reserved0Mask)));
        }
    }
}
