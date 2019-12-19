#ifndef _REGION_LAYER
#define _REGION_LAYER

#include <stdint.h>

typedef struct {
    uint32_t crood_num;
    uint32_t landm_num;
    uint32_t cls_num;
    uint32_t in_w;
    uint32_t in_h;
    const float *anchor;
    uint32_t anchor_num;
    float obj_thresh;
    float nms_thresh;
    float *variances;
    const float *bbox_input;
    const float *landm_input;
    const float *clses_input;
} region_layer_t;

typedef struct {
    uint32_t max_num; // 最大box数目
    uint32_t row_idx; // box的索引
    uint32_t col_idx; // box中位置的索引
    uint32_t box_len; // 每个box的长度
    uint32_t in_w;
    uint32_t in_h;
    uint32_t cls_num;
    uint32_t landm_num;
    uint32_t crood_num;
    float obj_thresh;
    float nms_thresh;
    float *box;
} box_info_t;

typedef struct {
    float x;
    float y;
    float w;
    float h;
} box_t;

typedef struct {
    uint32_t idx;
    const float *box;
} sortable_idx_t;

typedef void (*callback_draw_box)(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2,
                                  uint32_t class, float prob, uint32_t *landmark,
                                  uint32_t landm_num);
void softmax(const float *input, int n, float *output);
void region_layer_draw_boxes(box_info_t *bx, callback_draw_box callback);
void region_layer_run(region_layer_t *rl, box_info_t *bx);
int region_layer_init(region_layer_t *rl, const float *anchor, uint32_t anchor_num,
                      uint32_t crood_num, uint32_t landm_num, uint32_t cls_num, uint32_t in_w,
                      uint32_t in_h, float obj_thresh, float nms_thresh, float *variances);

int boxes_info_init(region_layer_t *rl, box_info_t *bx, int max_num);
void boxes_info_reset(box_info_t *bx);

#endif // _REGION_LAYER
