/*
 * Copyright (c) 2025, VeriSilicon Holdings Co., Ltd. All rights reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stddef.h>
#include <stdint.h>
#include "vs_conf.h"
#include "soc_init.h"
#include "soc_sysctl.h"
#include "bsp.h"
#include "uart_printf.h"
#include "board.h"
#include "osal.h"
#include "vpi_error.h"
#include "main.h"
#include "constants.h"
#include "speech_task.h"
#include "algo_task.h"
#include "audio_task.h"
#include "ble_task.h"
#include "vpi_ble_app.h"


static void task_init_app(void *param)
{
    int ret;
    BoardDevice board_dev;

    ret = board_register(board_get_ops());
    ret = vsd_to_vpi(ret);
    if (ret != VPI_SUCCESS) {
        uart_printf("board register failed %d", ret);
        goto exit;
    }
    ret = board_init((void *)&board_dev);
    ret = vsd_to_vpi(ret);
    if (ret != VPI_SUCCESS) {
        uart_printf("board init failed %d", ret);
        goto exit;
    }
    if (board_dev.name) {
        uart_printf("Board: %s", board_dev.name);
    }

    uart_printf("Hello VeriHealthi!\r\n");
    speech_service_init();
    ret = vpi_ble_start();
    if (ret != VPI_SUCCESS)
    {
        uart_printf("ble start failed\r\n");
    }
    else
    {

        osal_create_task(ble_app_task_entry, "ble_test", 3072, 1, NULL);
    }

//    osal_create_task(task_sample, "task_sample", 512, 4, NULL);
    osal_create_task(algo_task_entry, "algo_task", ALGO_TASK_STACK_SIZE, 3, NULL);
    osal_create_task(speech_task_entry, "speech_task", SPEECH_TASK_STACK_SIZE, 3, NULL);
    osal_create_task(audio_task_entry, "audio_task", 512, 2, NULL);
    //osal_create_task(print_task_entry, "print_task", 512, 2, NULL);
//    speech_service_init();
//    ret = vpi_ble_start();
//    if (ret != VPI_SUCCESS)
//    {
//        uart_printf("ble start failed\r\n");
//    }
//    else
//    {
//
//        osal_create_task(ble_app_task_entry, "ble_test", 3072, 1, NULL);
//    }
exit:
    osal_delete_task(NULL);
}

int main(void)
{
    int ret;

    ret = soc_init();
    ret = vsd_to_vpi(ret);
    if (ret != VPI_SUCCESS) {
        uart_printf("soc init error %d", ret);
        goto exit;
    } else {
        uart_printf("soc init done done");
    }
    osal_pre_start_scheduler();
    osal_create_task(task_init_app, "init_app", 512, 1, NULL);

//    osal_create_task(audio_task_entry, "speech_task", SPEECH_TASK_STACK_SIZE, 3, NULL);



    osal_start_scheduler();
exit:
    goto exit;
}
