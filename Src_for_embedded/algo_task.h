/*
 * algo_task.h
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */

#ifndef GALAXY_SDK_ALGO_TASK_H_
#define GALAXY_SDK_ALGO_TASK_H_

#include "constants.h"
#include "FreeRTOS.h"
#include "task.h"
#include "vpi_event.h"
#include "ble_task.h"

extern TaskHandle_t algo_task_handle;
void algo_task_entry(void *param);
void print_task_entry(void *param);



#endif /* GALAXY_SDK_ALGO_TASK_H_ */
