using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct kpu_batchnorm_argument_t
    {
        public ulong Value;

        private const int norm_mulShift = 0;
        private const ulong norm_mulMask = unchecked((ulong)((1UL << 24) - (1UL << 0)));
        public Bit24 norm_mul
        {
            get => (Bit24)((Value & norm_mulMask) >> norm_mulShift);
            set => Value = unchecked((ulong)((Value & ~norm_mulMask) | ((((ulong)value) << norm_mulShift) & norm_mulMask)));
        }
        private const int norm_addShift = 24;
        private const ulong norm_addMask = unchecked((ulong)((1UL << 56) - (1UL << 24)));
        public Bit32 norm_add
        {
            get => (Bit32)((Value & norm_addMask) >> norm_addShift);
            set => Value = unchecked((ulong)((Value & ~norm_addMask) | ((((ulong)value) << norm_addShift) & norm_addMask)));
        }
        private const int norm_shiftShift = 56;
        private const ulong norm_shiftMask = unchecked((ulong)((1UL << 60) - (1UL << 56)));
        public Bit4 norm_shift
        {
            get => (Bit4)((Value & norm_shiftMask) >> norm_shiftShift);
            set => Value = unchecked((ulong)((Value & ~norm_shiftMask) | ((((ulong)value) << norm_shiftShift) & norm_shiftMask)));
        }
    }
}
