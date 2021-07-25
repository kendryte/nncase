#ifndef _BOARD_CONFIG_
#define _BOARD_CONFIG_

#define  OV5640             0
#define  OV2640             1

#define  BOARD_KD233        1
#define  BOARD_LICHEEDAN    0

#if OV5640 + OV2640 != 1
#error ov sensor only choose one
#endif

#if BOARD_KD233 + BOARD_LICHEEDAN != 1
#error board only choose one
#endif

#endif
