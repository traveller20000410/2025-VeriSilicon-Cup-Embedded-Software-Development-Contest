/*
 * audio_task.h
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */

#ifndef GALAXY_SDK_AUDIO_TASK_H_
#define GALAXY_SDK_AUDIO_TASK_H_

#include "osal.h"
#include "hal_i2s.h"
#include "hal_codec.h"
#include "vpi_error.h"
#include "vsd_error.h"
#include "notify_start.h"
#include "notify_end.h"
#include "constants.h"
#include "FreeRTOS.h"
#include "task.h"
#include <string.h>
#include <stdio.h>
#include "speech_task.h"

#define AUDIO_TASK_STACK_SIZE   (1024 * 2) // 为audio_task定义2KB栈空间

/**
 * @brief audio_task的任务入口函数
 * @param param 任务创建时传入的参数
 */
void audio_task_entry(void *param);

/**
 * @brief 播放一段PCM数据
 *        注意：这是一个简化的阻塞式播放示例。
 *
 * @param data 指向PCM数据的指针
 * @param size PCM数据的字节数
 * @return int 0表示成功，非0表示失败
 */
int audio_task_play_pcm(const uint8_t *data, uint32_t size);

#endif /* GALAXY_SDK_AUDIO_TASK_H_ */
