/*
 * speech_task.h
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */

#ifndef GALAXY_SDK_SPEECH_TASK_H_
#define GALAXY_SDK_SPEECH_TASK_H_

#include "osal.h"
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>
#include "constants.h"
#include "FreeRTOS.h"
#include "task.h"
#include "event_groups.h"
#include "hal_pdm.h"
#include "vpi_event.h"


extern TaskHandle_t speech_task_handle;
extern CircularBuffer speech_pcm_circular_buffer;
void speech_task_entry(void *param);



#endif /* GALAXY_SDK_SPEECH_TASK_H_ */
