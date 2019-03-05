using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct conv_value2_t
    {
        public ulong Value;

        private const int arg_addShift = 0;
        private const ulong arg_addMask = unchecked((ulong)((1UL << 40) - (1UL << 0)));
        public Bit40 arg_add
        {
            get => (Bit40)((Value & arg_addMask) >> arg_addShift);
            set => Value = unchecked((ulong)((Value & ~arg_addMask) | ((((ulong)value) << arg_addShift) & arg_addMask)));
        }
        private const int reservedShift = 40;
        private const ulong reservedMask = unchecked((ulong)((1UL << 64) - (1UL << 40)));
        public Bit24 reserved
        {
            get => (Bit24)((Value & reservedMask) >> reservedShift);
            set => Value = unchecked((ulong)((Value & ~reservedMask) | ((((ulong)value) << reservedShift) & reservedMask)));
        }
    }
}
