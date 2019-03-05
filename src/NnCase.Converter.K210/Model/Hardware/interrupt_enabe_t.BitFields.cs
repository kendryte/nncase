using BitFields;

namespace NnCase.Converter.K210.Model.Hardware
{
    partial struct interrupt_enabe_t
    {
        public ulong Value;

        private const int int_enShift = 0;
        private const ulong int_enMask = unchecked((ulong)((1UL << 1) - (1UL << 0)));
        public Bit1 int_en
        {
            get => (Bit1)((Value & int_enMask) >> int_enShift);
            set => Value = unchecked((ulong)((Value & ~int_enMask) | ((((ulong)value) << int_enShift) & int_enMask)));
        }
        private const int ram_flagShift = 1;
        private const ulong ram_flagMask = unchecked((ulong)((1UL << 2) - (1UL << 1)));
        public Bit1 ram_flag
        {
            get => (Bit1)((Value & ram_flagMask) >> ram_flagShift);
            set => Value = unchecked((ulong)((Value & ~ram_flagMask) | ((((ulong)value) << ram_flagShift) & ram_flagMask)));
        }
        private const int full_addShift = 2;
        private const ulong full_addMask = unchecked((ulong)((1UL << 3) - (1UL << 2)));
        public Bit1 full_add
        {
            get => (Bit1)((Value & full_addMask) >> full_addShift);
            set => Value = unchecked((ulong)((Value & ~full_addMask) | ((((ulong)value) << full_addShift) & full_addMask)));
        }
        private const int depth_wise_layerShift = 3;
        private const ulong depth_wise_layerMask = unchecked((ulong)((1UL << 4) - (1UL << 3)));
        public Bit1 depth_wise_layer
        {
            get => (Bit1)((Value & depth_wise_layerMask) >> depth_wise_layerShift);
            set => Value = unchecked((ulong)((Value & ~depth_wise_layerMask) | ((((ulong)value) << depth_wise_layerShift) & depth_wise_layerMask)));
        }
        private const int reservedShift = 4;
        private const ulong reservedMask = unchecked((ulong)((1UL << 64) - (1UL << 4)));
        public Bit60 reserved
        {
            get => (Bit60)((Value & reservedMask) >> reservedShift);
            set => Value = unchecked((ulong)((Value & ~reservedMask) | ((((ulong)value) << reservedShift) & reservedMask)));
        }
    }
}
