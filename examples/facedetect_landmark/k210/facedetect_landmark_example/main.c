#include "board_config.h"
#include "bsp.h"
#include "dvp.h"
#include "fpioa.h"
#include "gpiohs.h"
#include "image_process.h"
#include "kpu.h"
#include "lcd.h"
#include "nt35310.h"
#include "ov2640.h"
#include "ov5640.h"
#include "plic.h"
#include "prior.h"
#include "region_layer.h"
#include "sysctl.h"
#include "uarths.h"
#include "utils.h"
#include "w25qxx.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#define INCBIN_PREFIX
#include "incbin.h"
#include "iomem.h"
#include "utils.h"

#define PLL0_OUTPUT_FREQ 800000000UL
#define PLL1_OUTPUT_FREQ 400000000UL

#define CLASS_NUMBER 1

volatile uint32_t g_ai_done_flag;
volatile uint8_t g_dvp_finish_flag;
static image_t kpu_image, display_image;
static region_layer_t rl;
static box_info_t boxes;
static float variances[2]= {0.1, 0.2};
float *pred_box, *pred_landm, *pred_clses;
size_t pred_box_size, pred_landm_size, pred_clses_size;

#define LOAD_KMODEL_FROM_FLASH 0

#if LOAD_KMODEL_FROM_FLASH
#define KMODEL_SIZE (412 * 1024)
uint8_t model_data[KMODEL_SIZE];
#else
INCBIN(model, "ulffd_landmark.kmodel");
#endif
kpu_model_context_t task;

static int ai_done(void *ctx) {
    g_ai_done_flag= 1;
    return 0;
}

static int dvp_irq(void *ctx) {
    if (dvp_get_interrupt(DVP_STS_FRAME_FINISH)) {
        dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 0);
        dvp_clear_interrupt(DVP_STS_FRAME_FINISH);
        g_dvp_finish_flag= 1;
    } else {
        dvp_start_convert();
        dvp_clear_interrupt(DVP_STS_FRAME_START);
    }
    return 0;
}

static void io_mux_init(void) {
#if BOARD_LICHEEDAN
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
#else
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
    fpioa_set_function(8, FUNC_GPIOHS0 + 2);
    fpioa_set_function(6, FUNC_SPI0_SS3);
    fpioa_set_function(7, FUNC_SPI0_SCLK);

    sysctl_set_spi0_dvp_data(1);
    fpioa_set_function(26, FUNC_GPIOHS0 + 8);
    gpiohs_set_drive_mode(8, GPIO_DM_INPUT);
#endif
}

static void io_set_power(void) {
#if BOARD_LICHEEDAN
    /* Set dvp and spi pin to 1.8V */
    sysctl_set_power_mode(SYSCTL_POWER_BANK6, SYSCTL_POWER_V18);
    sysctl_set_power_mode(SYSCTL_POWER_BANK7, SYSCTL_POWER_V18);
#else
    /* Set dvp and spi pin to 1.8V */
    sysctl_set_power_mode(SYSCTL_POWER_BANK0, SYSCTL_POWER_V18);
    sysctl_set_power_mode(SYSCTL_POWER_BANK1, SYSCTL_POWER_V18);
    sysctl_set_power_mode(SYSCTL_POWER_BANK2, SYSCTL_POWER_V18);
#endif
}

#if (CLASS_NUMBER > 1)
typedef struct {
    char *str;
    uint16_t color;
    uint16_t height;
    uint16_t width;
    uint32_t *ptr;
} class_lable_t;

class_lable_t class_lable[CLASS_NUMBER]= {
    {"aeroplane", GREEN}, {"bicycle", GREEN},   {"bird", GREEN},        {"boat", GREEN},
    {"bottle", 0xF81F},   {"bus", GREEN},       {"car", GREEN},         {"cat", GREEN},
    {"chair", 0xFD20},    {"cow", GREEN},       {"diningtable", GREEN}, {"dog", GREEN},
    {"horse", GREEN},     {"motorbike", GREEN}, {"person", 0xF800},     {"pottedplant", GREEN},
    {"sheep", GREEN},     {"sofa", GREEN},      {"train", GREEN},       {"tvmonitor", 0xF9B6}};

static uint32_t lable_string_draw_ram[115 * 16 * 8 / 2];
#endif

static void lable_init(void) {
#if (CLASS_NUMBER > 1)
    uint8_t index;

    class_lable[0].height= 16;
    class_lable[0].width= 8 * strlen(class_lable[0].str);
    class_lable[0].ptr= lable_string_draw_ram;
    lcd_ram_draw_string(class_lable[0].str, class_lable[0].ptr, BLACK, class_lable[0].color);
    for (index= 1; index < CLASS_NUMBER; index++) {
        class_lable[index].height= 16;
        class_lable[index].width= 8 * strlen(class_lable[index].str);
        class_lable[index].ptr= class_lable[index - 1].ptr +
                                class_lable[index - 1].height * class_lable[index - 1].width / 2;
        lcd_ram_draw_string(class_lable[index].str, class_lable[index].ptr, BLACK,
                            class_lable[index].color);
    }
#endif
}

static void drawboxes(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t class,
                      float prob, uint32_t *landmark, uint32_t landm_num) {
    if (x1 >= 320) x1= 319;
    if (x2 >= 320) x2= 319;
    if (y1 >= 224) y1= 223;
    if (y2 >= 224) y2= 223;

#if (CLASS_NUMBER > 1)
    lcd_draw_rectangle(x1, y1, x2, y2, 2, class_lable[class].color);
    lcd_draw_picture(x1 + 1, y1 + 1, class_lable[class].width, class_lable[class].height,
                     class_lable[class].ptr);
#else
    lcd_draw_rectangle(x1, y1, x2, y2, 2, RED);
    for (uint32_t i= 0; i < landm_num; i++) {
        lcd_draw_point(landmark[2 * i], landmark[1 + 2 * i], GREEN);
    }
#endif
}

int main(void) {
    /* Set CPU and dvp clk */
    sysctl_pll_set_freq(SYSCTL_PLL0, PLL0_OUTPUT_FREQ);
    sysctl_pll_set_freq(SYSCTL_PLL1, PLL1_OUTPUT_FREQ);
    sysctl_clock_enable(SYSCTL_CLOCK_AI);
    uarths_init();
    io_set_power();
    io_mux_init();
    plic_init();

    // lable_init();

#if LOAD_KMODEL_FROM_FLASH
    /* flash init */
    printf("flash init\n");
    w25qxx_init(3, 0);
    w25qxx_enable_quad_mode();
    w25qxx_read_data(0xA00000, model_data, KMODEL_SIZE, W25QXX_QUAD_FAST);
#endif
    /* LCD init */
    printf("LCD init\n");
    lcd_init();
#if BOARD_LICHEEDAN
#if OV5640
    lcd_set_direction(DIR_YX_RLUD);
#else
    lcd_set_direction(DIR_YX_RLDU);
#endif
#else
#if OV5640
    lcd_set_direction(DIR_YX_RLUD);
#else
    lcd_set_direction(DIR_YX_LRDU);
#endif
#endif
    lcd_clear(BLACK);
    /* DVP init */
    printf("DVP init\n");
#if OV5640
    dvp_init(16);
    dvp_set_xclk_rate(12000000);
    dvp_enable_burst();
    dvp_set_output_enable(0, 1);
    dvp_set_output_enable(1, 1);
    dvp_set_image_format(DVP_CFG_RGB_FORMAT);
    dvp_set_image_size(320, 240);
    ov5640_init();
#else
    dvp_init(8);
    dvp_set_xclk_rate(24000000);
    dvp_enable_burst();
    dvp_set_output_enable(0, 1);
    dvp_set_output_enable(1, 1);
    dvp_set_image_format(DVP_CFG_RGB_FORMAT);
    dvp_set_image_size(320, 224);
    ov2640_init();
#endif
    kpu_image.pixel= 3;
    kpu_image.width= 320;
    kpu_image.height= 240;
    image_init(&kpu_image);
    display_image.pixel= 2;
    display_image.width= 320;
    display_image.height= 240;
    image_init(&display_image);
    dvp_set_ai_addr((uint32_t)kpu_image.addr, (uint32_t)(kpu_image.addr + 320 * 240),
                    (uint32_t)(kpu_image.addr + 320 * 240 * 2));
    dvp_set_display_addr((uint32_t)display_image.addr);
    dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 0);
    dvp_disable_auto();
    /* DVP interrupt config */
    printf("DVP interrupt config\n");
    plic_set_priority(IRQN_DVP_INTERRUPT, 1);
    plic_irq_register(IRQN_DVP_INTERRUPT, dvp_irq, NULL);
    plic_irq_enable(IRQN_DVP_INTERRUPT);
    /* init face detect model */
    if (kpu_load_kmodel(&task, model_data) != 0) {
        printf("\nmodel init error\n");
        while (1) {};
    }

    region_layer_init(&rl, anchor, 3160, 4, 5, 1, 320, 240, 0.7, 0.4, variances);
    boxes_info_init(&rl, &boxes, 200);

    /* enable global interrupt */
    sysctl_enable_irq();
    /* system start */
    printf("System start\n");

    while (1) {
        g_dvp_finish_flag= 0;
        dvp_clear_interrupt(DVP_STS_FRAME_START | DVP_STS_FRAME_FINISH);
        dvp_config_interrupt(DVP_CFG_START_INT_ENABLE | DVP_CFG_FINISH_INT_ENABLE, 1);

        while (g_dvp_finish_flag == 0) {};
        /* run face detect */
        g_ai_done_flag= 0;
        kpu_run_kmodel(&task, (uint8_t *)kpu_image.addr, DMAC_CHANNEL5, ai_done, NULL);
        while (!g_ai_done_flag) {};

        kpu_get_output(&task, 0, (uint8_t **)&pred_box, &pred_box_size);
        kpu_get_output(&task, 1, (uint8_t **)&pred_landm, &pred_landm_size);
        kpu_get_output(&task, 2, (uint8_t **)&pred_clses, &pred_clses_size);

        rl.bbox_input= pred_box;
        rl.landm_input= pred_landm;
        rl.clses_input= pred_clses;
        region_layer_run(&rl, &boxes);

        // do_more_nms_sort(&detect_rl0, &detect_rl1);
        /* display result */
        // rgb888_to_lcd(ai_image.addr, lcd_gram, 320, 224);
        // lcd_draw_picture(0, 0, 320, 224, lcd_gram);
        lcd_draw_picture(0, 0, 320, 224, (uint32_t *)display_image.addr);

        /* run key point detect */
        region_layer_draw_boxes(&boxes, drawboxes);

        boxes_info_reset(&boxes);
    }
}
