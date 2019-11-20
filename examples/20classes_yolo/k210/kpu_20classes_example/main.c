#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include "fpioa.h"
#include "lcd.h"
#include "board_config.h"
#include "plic.h"
#include "sysctl.h"
#include "uarths.h"
#include "nt35310.h"
#include "utils.h"
#include "kpu.h"
#include "region_layer.h"
#include "w25qxx.h"
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"

#define PLL0_OUTPUT_FREQ 800000000UL
#define PLL1_OUTPUT_FREQ 400000000UL

#define CLASS_NUMBER 20

#define LOAD_KMODEL_FROM_FLASH 0

extern const unsigned char gImage_image[] __attribute__((aligned(128)));

#if LOAD_KMODEL_FROM_FLASH
#define KMODEL_SIZE (1351976)
uint8_t *model_data;
#else
INCBIN(model, "yolo.kmodel");
#endif
kpu_model_context_t task;
static region_layer_t detect_rl;

volatile uint8_t g_ai_done_flag;
static uint16_t lcd_gram[320 * 240] __attribute__((aligned(32)));

static int ai_done(void *ctx)
{
    g_ai_done_flag = 1;
    return 0;
}

#define ANCHOR_NUM 5

float g_anchor[ANCHOR_NUM * 2] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};

#if BOARD_LICHEEDAN
static void io_mux_init(void)
{
    /* Init DVP IO map and function settings */
    fpioa_set_function(42, FUNC_CMOS_RST);
    fpioa_set_function(44, FUNC_CMOS_PWDN);
    fpioa_set_function(46, FUNC_CMOS_XCLK);
    fpioa_set_function(43, FUNC_CMOS_VSYNC);
    fpioa_set_function(45, FUNC_CMOS_HREF);
    fpioa_set_function(47, FUNC_CMOS_PCLK);
    fpioa_set_function(41, FUNC_SCCB_SCLK);
    fpioa_set_function(40, FUNC_SCCB_SDA);

    /* Init SPI IO map and function settings */
    fpioa_set_function(38, FUNC_GPIOHS0 + DCX_GPIONUM);
    fpioa_set_function(36, FUNC_SPI0_SS3);
    fpioa_set_function(39, FUNC_SPI0_SCLK);
    fpioa_set_function(37, FUNC_GPIOHS0 + RST_GPIONUM);

    sysctl_set_spi0_dvp_data(1);
}

static void io_set_power(void)
{
    /* Set dvp and spi pin to 1.8V */
    sysctl_set_power_mode(SYSCTL_POWER_BANK6, SYSCTL_POWER_V18);
    sysctl_set_power_mode(SYSCTL_POWER_BANK7, SYSCTL_POWER_V18);
}

#else
static void io_mux_init(void)
{
    /* Init DVP IO map and function settings */
    fpioa_set_function(11, FUNC_CMOS_RST);
    fpioa_set_function(13, FUNC_CMOS_PWDN);
    fpioa_set_function(14, FUNC_CMOS_XCLK);
    fpioa_set_function(12, FUNC_CMOS_VSYNC);
    fpioa_set_function(17, FUNC_CMOS_HREF);
    fpioa_set_function(15, FUNC_CMOS_PCLK);
    fpioa_set_function(10, FUNC_SCCB_SCLK);
    fpioa_set_function(9, FUNC_SCCB_SDA);

    /* Init SPI IO map and function settings */
    fpioa_set_function(8, FUNC_GPIOHS0 + DCX_GPIONUM);
    fpioa_set_function(6, FUNC_SPI0_SS3);
    fpioa_set_function(7, FUNC_SPI0_SCLK);

    sysctl_set_spi0_dvp_data(1);
}

static void io_set_power(void)
{
    /* Set dvp and spi pin to 1.8V */
    sysctl_set_power_mode(SYSCTL_POWER_BANK1, SYSCTL_POWER_V18);
    sysctl_set_power_mode(SYSCTL_POWER_BANK2, SYSCTL_POWER_V18);
}
#endif

#if (CLASS_NUMBER > 1)
typedef struct
{
    char *str;
    uint16_t color;
    uint16_t height;
    uint16_t width;
    uint32_t *ptr;
} class_lable_t;

class_lable_t class_lable[CLASS_NUMBER] =
{
    {"aeroplane", GREEN},
    {"bicycle", GREEN},
    {"bird", GREEN},
    {"boat", GREEN},
    {"bottle", 0xF81F},
    {"bus", GREEN},
    {"car", GREEN},
    {"cat", GREEN},
    {"chair", 0xFD20},
    {"cow", GREEN},
    {"diningtable", GREEN},
    {"dog", GREEN},
    {"horse", GREEN},
    {"motorbike", GREEN},
    {"person", 0xF800},
    {"pottedplant", GREEN},
    {"sheep", GREEN},
    {"sofa", GREEN},
    {"train", GREEN},
    {"tvmonitor", 0xF9B6}
};

static uint32_t lable_string_draw_ram[115 * 16 * 8 / 2];
#endif

static void lable_init(void)
{
#if (CLASS_NUMBER > 1)
    uint8_t index;

    class_lable[0].height = 16;
    class_lable[0].width = 8 * strlen(class_lable[0].str);
    class_lable[0].ptr = lable_string_draw_ram;
    lcd_ram_draw_string(class_lable[0].str, class_lable[0].ptr, BLACK, class_lable[0].color);
    for (index = 1; index < CLASS_NUMBER; index++) {
        class_lable[index].height = 16;
        class_lable[index].width = 8 * strlen(class_lable[index].str);
        class_lable[index].ptr = class_lable[index - 1].ptr + class_lable[index - 1].height * class_lable[index - 1].width / 2;
        lcd_ram_draw_string(class_lable[index].str, class_lable[index].ptr, BLACK, class_lable[index].color);
    }
#endif
}

static void drawboxes(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t class, float prob)
{
    if (x1 >= 320)
        x1 = 319;
    if (x2 >= 320)
        x2 = 319;
    if (y1 >= 240)
        y1 = 239;
    if (y2 >= 240)
        y2 = 239;

#if (CLASS_NUMBER > 1)
    lcd_draw_rectangle(x1, y1, x2, y2, 2, class_lable[class].color);
    lcd_draw_picture(x1 + 1, y1 + 1, class_lable[class].width, class_lable[class].height, class_lable[class].ptr);
#else
    lcd_draw_rectangle(x1, y1, x2, y2, 2, RED);
#endif
}

void rgb888_to_lcd(uint8_t *src, uint16_t *dest, size_t width, size_t height)
{
    size_t i, chn_size = width * height;
    for (size_t i = 0; i < width * height; i++)
    {
        uint8_t r = src[i];
        uint8_t g = src[chn_size + i];
        uint8_t b = src[chn_size * 2 + i];

        uint16_t rgb = ((r & 0b11111000) << 8) | ((g & 0b11111100) << 3) | (b >> 3);
        size_t d_i = i % 2 ? (i - 1) : (i + 1);
        dest[d_i] = rgb;
    }
}

int main(void)
{
    /* Set CPU and dvp clk */
    sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
    sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
    uarths_init();

    io_mux_init();
    io_set_power();
    plic_init();

#if LOAD_KMODEL_FROM_FLASH
    printf("flash init\n");
    w25qxx_init(3, 0);
    w25qxx_enable_quad_mode();
    model_data = (uint8_t *)malloc(KMODEL_SIZE);
    w25qxx_read_data(0xA00000, model_data, KMODEL_SIZE, W25QXX_QUAD_FAST);
#endif
    lable_init();

    /* LCD init */
    printf("LCD init\n");
    lcd_init();
#if BOARD_LICHEEDAN
    lcd_set_direction(DIR_YX_RLDU);
#else
    lcd_set_direction(DIR_YX_RLUD);
#endif
    /* enable global interrupt */
    sysctl_enable_irq();

    /* system start */
    printf("system start\n");

    /* init kpu */
    if (kpu_load_kmodel(&task, model_data) != 0)
    {
        printf("\nmodel init error\n");
        while (1);
    }

    detect_rl.anchor_number = ANCHOR_NUM;
    detect_rl.anchor = g_anchor;
    detect_rl.threshold = 0.5;
    detect_rl.nms_value = 0.2;
    region_layer_init(&detect_rl, 10, 7, 125, 320, 240);

    {
        g_ai_done_flag = 0;
        /* start to calculate */
        kpu_run_kmodel(&task, gImage_image, DMAC_CHANNEL5, ai_done, NULL);
        while(!g_ai_done_flag);

        float *output;
        size_t output_size;
        kpu_get_output(&task, 0, &output, &output_size);
        detect_rl.input = output;
        output_size /= sizeof(float);

        /* start region layer */
        region_layer_run(&detect_rl, NULL);

        /* display pic*/
        rgb888_to_lcd(gImage_image, lcd_gram, 320, 240);
        lcd_draw_picture(0, 0, 320, 240, lcd_gram);

        /* draw boxs */
        region_layer_draw_boxes(&detect_rl, drawboxes);
    }

    while(1);
}

