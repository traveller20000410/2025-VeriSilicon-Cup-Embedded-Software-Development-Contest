/*
 * conv.h
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */

#ifndef GALAXY_SDK_CONV_H_
#define GALAXY_SDK_CONV_H_

#include "constants.h"
#include "math.h"

typedef struct _Conv2dData {
    uint16_t row;
    uint16_t col;
    uint16_t channel;
    float *data;
} Conv2dData;

typedef struct _Conv2dFilter {
    uint16_t row;
    uint16_t col;
    uint16_t channel;
    uint16_t filter_num;
    const float *data;
} Conv2dFilter;

typedef struct _BatchNorm2d {
    uint16_t size;
    const float *mean;
    const float *var;
    const float *gamma;
    const float *beta;
} BatchNorm2d;

typedef struct _Conv2dConfig {
    uint16_t stride;
    uint16_t pad;
    Conv2dFilter *filter;
    BatchNorm2d *bn;
} Conv2dConfig;

typedef struct _LinearConfig {
    uint16_t inp_size;
    uint16_t fea_size;
    const float *weight;
    const float *bias;
} LinearParam;

uint16_t cal_conv_out_len(uint16_t raw_len, uint16_t pad_len, uint16_t filter_len, uint16_t stride);
int conv2d_bn_no_bias(Conv2dData *input_feat, Conv2dConfig *param, Conv2dData *output_feat);
int leaky_relu(float neg_slope, float *inp, uint16_t inp_size, float *out);
int linear_layer(float *inp, LinearParam *linear_config, float *out);
int max_pool2d(const Conv2dData *input_feat, Conv2dData *output_feat, uint16_t kernel_size, uint16_t stride);



#endif /* GALAXY_SDK_CONV_H_ */
