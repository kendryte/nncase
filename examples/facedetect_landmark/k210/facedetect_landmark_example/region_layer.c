#include "region_layer.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float overlap(const float x1, const float w1, const float x2, const float w2) {
    float l1= x1 - w1 / 2;
    float l2= x2 - w2 / 2;
    float left= l1 > l2 ? l1 : l2;
    float r1= x1 + w1 / 2;
    float r2= x2 + w2 / 2;
    float right= r1 < r2 ? r1 : r2;

    return right - left;
}

static float box_intersection(const box_t *a, const box_t *b) {
    float w= overlap(a->x, a->w, b->x, b->w);
    float h= overlap(a->y, a->h, b->y, b->h);

    if (w < 0 || h < 0) return 0;
    return w * h;
}

static float box_union(const box_t *a, const box_t *b) {
    float i= box_intersection(a, b);
    float u= a->w * a->h + b->w * b->h - i;

    return u;
}

static float box_iou(const box_t *a, const box_t *b) {
    return box_intersection(a, b) / box_union(a, b);
}

static inline float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

void softmax(const float *input, int n, float *output) {
    int i;
    float diff;
    float e;
    float sum= 0;
    float largest_i= input[0];

    for (i= 0; i < n; ++i) {
        if (input[i] > largest_i) { largest_i= input[i]; }
    }

    for (i= 0; i < n; ++i) {
        diff= input[i] - largest_i;
        e= expf(diff);
        sum+= e;
        output[i]= e;
    }
    for (i= 0; i < n; ++i) { output[i]/= sum; }
}

static void forward_region_layer(region_layer_t *rl, box_info_t *bx) {
    float probs[2];
    float *box;
    const float *pred_box, *landm, *anc;
    for (uint32_t i= 0; i < rl->anchor_num; i++) {
        softmax(&rl->clses_input[i * 2], 2, probs);
        if (probs[1] > rl->obj_thresh) { // probs[1] is pos prob
            // find index
            box= &bx->box[bx->row_idx * bx->box_len];
            pred_box= &rl->bbox_input[i * rl->crood_num];
            landm= &rl->landm_input[i * rl->landm_num * 2];
            anc= &rl->anchor[i * rl->crood_num];

            // decode pred_box, output pred_box is x,y,w,h
            box[0]= anc[0] + pred_box[0] * rl->variances[0] * anc[2];
            box[1]= anc[1] + pred_box[1] * rl->variances[0] * anc[3];
            box[2]= anc[2] * expf(pred_box[2] * rl->variances[1]);
            box[3]= anc[3] * expf(pred_box[3] * rl->variances[1]);
            // pos prob
            box[4]= probs[1];
            // decode landm
            for (uint32_t j= 0; j < rl->landm_num; j++) {
                box[5 + j * 2]= anc[0] + landm[j * 2] * rl->variances[0] * anc[2];
                box[6 + j * 2]= anc[1] + landm[j * 2 + 1] * rl->variances[0] * anc[3];
            }
            bx->row_idx++;
            if (bx->row_idx == bx->max_num) { return; }
        }
    }
}

static int nms_comparator(const void *pa, const void *pb) {
    const sortable_idx_t l= *(const sortable_idx_t *)pa;
    const sortable_idx_t r= *(const sortable_idx_t *)pb;
    float diff= l.box[l.idx] - r.box[r.idx];

    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

static void do_nms_sort(const box_info_t *bx) {
    sortable_idx_t sort_idx[bx->row_idx];
    float *box= bx->box;
    for (uint32_t i= 0; i < bx->cls_num; i++) {
        for (uint32_t j= 0; j < bx->row_idx; j++) {
            sort_idx[j].box= box;
            sort_idx[j].idx= bx->box_len * j + bx->crood_num + i;
        }
        // sort by high => low
        qsort(sort_idx, bx->row_idx, sizeof(sortable_idx_t),
              (int (*)(const void *, const void *))nms_comparator);

        for (uint32_t k= 0; k < bx->row_idx; k++) {
            if (box[sort_idx[k].idx] == 0) { continue; }
            box_t *a= (box_t *)&box[(sort_idx[k].idx / bx->box_len) * bx->box_len];

            for (uint32_t q= k + 1; q < bx->row_idx; q++) {
                box_t *b= (box_t *)&box[(sort_idx[q].idx / bx->box_len) * bx->box_len];
                if (box_iou(a, b) > bx->nms_thresh) { box[sort_idx[q].idx]= 0; }
            }
        }
    }
}

void region_layer_draw_boxes(box_info_t *bx, callback_draw_box callback) {
    uint32_t landmark[bx->landm_num * 2];
    float prob;
    for (uint32_t i= 0; i < bx->row_idx; i++) {
        prob= bx->box[bx->box_len * i + bx->crood_num];
        if (prob > bx->obj_thresh) {
            float *b= &bx->box[bx->box_len * i];
            uint32_t x1= (b[0] - (b[2] / 2)) * bx->in_w;
            uint32_t y1= (b[1] - (b[3] / 2)) * bx->in_h;
            uint32_t x2= (b[0] + (b[2] / 2)) * bx->in_w;
            uint32_t y2= (b[1] + (b[3] / 2)) * bx->in_h;
            // printf("%d\t%d\t%d\t%d\t%f\t", x1, y1, x2, y2, b[4]);
            for (uint32_t j= 0; j < bx->landm_num; j++) {
                landmark[0 + j * 2]= (uint32_t)(b[5 + j * 2] * bx->in_w);
                landmark[1 + j * 2]= (uint32_t)(b[6 + j * 2] * bx->in_h);
            }
            callback(x1, y1, x2, y2, i, prob, landmark, bx->landm_num);
        }
    }
}

void region_layer_run(region_layer_t *rl, box_info_t *bx) {
    forward_region_layer(rl, bx);
    do_nms_sort(bx);
}

int region_layer_init(region_layer_t *rl, const float *anchor, uint32_t anchor_num,
                      uint32_t crood_num, uint32_t landm_num, uint32_t cls_num, uint32_t in_w,
                      uint32_t in_h, float obj_thresh, float nms_thresh, float *variances) {
    rl->crood_num= crood_num;
    rl->landm_num= landm_num;
    rl->cls_num= cls_num;
    rl->in_w= in_w;
    rl->in_h= in_h;
    rl->anchor= anchor;
    rl->anchor_num= anchor_num;
    rl->obj_thresh= obj_thresh;
    rl->nms_thresh= nms_thresh;
    rl->variances= variances;
    return 0;
}

int boxes_info_init(region_layer_t *rl, box_info_t *bx, int max_num) {
    bx->in_w= rl->in_w;
    bx->in_h= rl->in_h;
    bx->cls_num= rl->cls_num;
    bx->landm_num= rl->landm_num;
    bx->crood_num= rl->crood_num;
    bx->obj_thresh= rl->obj_thresh;
    bx->nms_thresh= rl->nms_thresh;
    bx->row_idx= 0;
    bx->col_idx= 0;
    // NOTE each bbox
    bx->box_len= rl->crood_num + 1 + rl->landm_num * 2;
    bx->max_num= max_num;
    bx->box= malloc(sizeof(float) * bx->box_len * bx->max_num);
    if (bx->box == NULL) { return -1; }
    // NOTE when infer once need reset
    memset(bx->box, 0, sizeof(float) * bx->box_len * bx->max_num);

    return 0;
}

void boxes_info_reset(box_info_t *bx) {
    bx->col_idx= 0;
    bx->row_idx= 0;
    memset(bx->box, 0, sizeof(float) * bx->box_len * bx->max_num);
}