/*
 * conv.c
 *
 *  Created on: 2025年7月23日
 *      Author: 柴焰旺
 */

#include <math.h>
#include "conv.h"
#include "riscv_math.h"   // NMSIS‑DSP

uint16_t cal_conv_out_len(uint16_t raw_len, uint16_t pad_len, uint16_t filter_len, uint16_t stride)
{
    return (raw_len + 2 * pad_len - filter_len) / stride + 1;
}

int conv2d_bn_no_bias(Conv2dData *input_feat, Conv2dConfig *param, Conv2dData *output_feat)
{
    if (!input_feat || !input_feat->data ||
        !param || !param->bn || !param->filter ||
        !output_feat || !output_feat->data)
    {
        return ALGO_POINTER_NULL;
    }

    BatchNorm2d   *bn     = param->bn;
    Conv2dFilter  *filter = param->filter;

    // 基本合法性检查
    if (param->stride < 1 ||
        filter->channel != input_feat->channel ||
        filter->filter_num != bn->size ||
        param->pad != 0)    // TODO: 支持非零 pad 时需要 im2col 填充
    {
        return ALGO_DATA_EXCEPTION;
    }

    // 计算输出尺寸
    uint16_t out_row  = input_feat->row == 1
                       ? 1
                       : cal_conv_out_len(input_feat->row, param->pad, filter->row, param->stride);
    uint16_t out_col  = cal_conv_out_len(input_feat->col, param->pad, filter->col, param->stride);
    uint16_t out_chan = filter->filter_num;

    uint16_t paded_row = input_feat->row + 2 * param->pad;
    uint16_t paded_col = input_feat->col + 2 * param->pad;
    const float *paded_feat = input_feat->data;

    // 卷积核元素数
    uint32_t K = filter->channel * filter->row * filter->col;
    // 用于 im2col 的临时缓冲区
    float32_t patch[K];

    for (uint16_t oc = 0; oc < out_chan; oc++)
    {
        // 预计算 BatchNorm scale 和 bias
        float32_t scale = bn->gamma[oc] / sqrtf(bn->var[oc] + BN_EPS);
        float32_t bias  = bn->beta[oc] - bn->mean[oc] * scale;

        // 当前输出通道对应的卷积核起始地址
        const float32_t *ker_base = filter->data + (uint32_t)oc * K;

        for (uint16_t orow = 0; orow < out_row; orow++)
        {
            uint16_t row_start = orow * param->stride;
            for (uint16_t ocol = 0; ocol < out_col; ocol++)
            {
                uint16_t col_start = ocol * param->stride;

                // im2col：将当前滑窗下的所有输入拆成一维 patch[]
                uint32_t idx = 0;
                for (uint16_t ic = 0; ic < filter->channel; ic++)
                {
                    uint32_t channel_offset = ic * paded_row * paded_col;
                    for (uint16_t kr = 0; kr < filter->row; kr++)
                    {
                        uint32_t base = channel_offset
                                      + (row_start + kr) * paded_col
                                      + col_start;
                        for (uint16_t kc = 0; kc < filter->col; kc++)
                        {
                            patch[idx++] = paded_feat[base + kc];
                        }
                    }
                }

                // 调用 NMSIS‑DSP 点积
                float32_t acc;
                riscv_dot_prod_f32(ker_base, patch, K, &acc);

                // 应用预计算的 scale 和 bias
                acc = acc * scale + bias;

                // 写回输出
                uint32_t out_index = (uint32_t)oc * out_row * out_col
                                   + (uint32_t)orow * out_col
                                   + ocol;
                output_feat->data[out_index] = acc;
            }
        }
    }

    output_feat->row     = out_row;
    output_feat->col     = out_col;
    output_feat->channel = out_chan;
    return ALGO_NORMAL;
}

int leaky_relu(float neg_slope, float *inp, uint16_t inp_size, float *out)
{
    if (!inp || !out) return ALGO_POINTER_NULL;
    for (uint16_t i = 0; i < inp_size; i++)
        out[i] = (inp[i] < 0) ? inp[i] * neg_slope : inp[i];
    return ALGO_NORMAL;
}

int linear_layer(float *inp, LinearParam *cfg, float *out)
{
    if (!inp || !cfg || !cfg->weight || !cfg->bias || !out)
        return ALGO_POINTER_NULL;

    const float *w = cfg->weight;
    for (uint16_t i = 0; i < cfg->fea_size; i++)
    {
        float sum = cfg->bias[i];
        for (uint16_t j = 0; j < cfg->inp_size; j++)
            sum += inp[j] * (*w++);
        out[i] = sum;
    }
    return ALGO_NORMAL;
}

int max_pool2d(const Conv2dData *input_feat,
               Conv2dData *output_feat,
               uint16_t kernel_size,
               uint16_t stride)
{
    if (!input_feat || !input_feat->data ||
        !output_feat || !output_feat->data)
        return ALGO_POINTER_NULL;
    if (kernel_size == 0 || stride == 0)
        return ALGO_DATA_EXCEPTION;

    uint16_t expected_row = cal_conv_out_len(input_feat->row, 0, kernel_size, stride);
    uint16_t expected_col = cal_conv_out_len(input_feat->col, 0, kernel_size, stride);
    if (output_feat->row != expected_row ||
        output_feat->col != expected_col ||
        output_feat->channel != input_feat->channel)
        return ALGO_DATA_EXCEPTION;
    if (expected_row == 0 || expected_col == 0)
        return ALGO_NORMAL;

    uint32_t in_ch_sz  = (uint32_t)input_feat->row * input_feat->col;
    uint32_t out_ch_sz = (uint32_t)output_feat->row * output_feat->col;

    for (uint16_t c = 0; c < input_feat->channel; c++)
    {
        uint32_t in_off  = c * in_ch_sz;
        uint32_t out_off = c * out_ch_sz;
        for (uint16_t orow = 0; orow < output_feat->row; orow++)
        {
            for (uint16_t ocol = 0; ocol < output_feat->col; ocol++)
            {
                uint16_t r0 = orow * stride;
                uint16_t c0 = ocol * stride;
                float   m   = input_feat->data[in_off + r0 * input_feat->col + c0];
                for (uint16_t kr = 0; kr < kernel_size; kr++)
                {
                    for (uint16_t kc = 0; kc < kernel_size; kc++)
                    {
                        float v = input_feat->data[in_off
                                + (r0 + kr) * input_feat->col
                                + (c0 + kc)];
                        if (v > m) m = v;
                    }
                }
                output_feat->data[out_off + orow * output_feat->col + ocol] = m;
            }
        }
    }
    return ALGO_NORMAL;
}
