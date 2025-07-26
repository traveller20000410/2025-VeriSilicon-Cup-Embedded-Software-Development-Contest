/*
 * speech_task.c
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */


#include "speech_task.h"
#include "FreeRTOS.h"
#include "task.h"
#include "hal_pdm.h"
#include "vpi_event.h"
#include "event_groups.h"
#include "algo_task.h"
#include "vpi_error.h"
#include <string.h>

TaskHandle_t speech_task_handle;
static PdmSubstream pdm_stream_config;
static PdmDevice *pdm_dev_handle = NULL;

//static uint8_t pdm_driver_rx_buffer[PDM_BUFFER_SIZE_BYTES];
static uint8_t pdm_driver_rx_buffer[8];
// 全局环形缓冲区实例和底层数据存储
// 缓冲5帧数据,cyw改成了1.5倍的长度
#define SPEECH_CIRCULAR_BUFFER_FRAME_COUNT 20
static uint8_t speech_pcm_buffer_data[PDM_BUFFER_SIZE_BYTES *
                                      SPEECH_CIRCULAR_BUFFER_FRAME_COUNT / 10];
CircularBuffer speech_pcm_circular_buffer;


//#define DUMP_BUFFER_SIZE 100000
//static uint8_t pdm_dump_buffer[DUMP_BUFFER_SIZE];
//
//// 2. 定义一个计数器，用于跟踪已写入dump缓冲区的字节数
//static volatile uint32_t dump_buffer_write_count = 0;


int test = 0;
void pdm_irq_handler_callback(const PdmDevice *pdm_device, int received_size,
                              void *cb_ctx)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    PdmSubstream *current_stream = (PdmSubstream *)cb_ctx;
    test= test+ 8;
    if (received_size > 0 && current_stream) {
        // 将PDM驱动接收到的数据写入环形缓冲区
        circularbuffer_write(&speech_pcm_circular_buffer,
                             current_stream->buffer.base,
                             received_size);

        // 如果环形缓冲区中有足够的数据（例如，一帧），则发送事件
        if (circularbuffer_getusedsize(&speech_pcm_circular_buffer) >=
            (PDM_BUFFER_SIZE_BYTES) &&test >= 8000) {
            // 不需要传递PcmDataInfo结构体，只需通知algo_task有数据可读
            vpi_event_notify_from_isr(EVT_PCM_DATA_READY, NULL);
            test = 0;
        }
    }
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);

//    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
//    PdmSubstream *current_stream = (PdmSubstream *)cb_ctx;
//
//    if (received_size > 0 && current_stream) {
//        // --- 【新的中断逻辑】---
//        // 检查dump缓冲区是否还有空间，并且断点还未触发
//        if ( dump_buffer_write_count + received_size <= DUMP_BUFFER_SIZE)
//        {
//            // 将本次DMA收到的数据，拷贝到我们的大缓冲区中
//            memcpy(&pdm_dump_buffer[dump_buffer_write_count],
//                   current_stream->buffer.base,
//                   received_size);
//
//            // 更新计数器
//            dump_buffer_write_count += received_size;
//        }
//
//        // 检查是否已存满
//        if ( dump_buffer_write_count >= DUMP_BUFFER_SIZE )
//        {
//            printf("ok");
//        }
//    }
//
//    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

void speech_task_entry(void *param)
{
    speech_task_handle = xTaskGetCurrentTaskHandle();

    pdm_stream_config.sample_rate = 8000;
    pdm_stream_config.sample_width = 16;
    pdm_stream_config.chan_num = 1;
    pdm_stream_config.xfer_mode = XFER_MODE_INTR;
    pdm_stream_config.cb = pdm_irq_handler_callback;
    pdm_stream_config.cb_context = (void *)&pdm_stream_config;

    // 初始化 PDM 数据缓冲区，供 PDM 驱动使用
    pdm_stream_config.buffer.base = pdm_driver_rx_buffer;
    pdm_stream_config.buffer.size = sizeof(pdm_driver_rx_buffer);

    // 初始化环形缓冲区
    circularbuffer_init(&speech_pcm_circular_buffer, speech_pcm_buffer_data,
                        sizeof(speech_pcm_buffer_data));

    pdm_dev_handle = hal_pdm_get_device(PDM_ID_0);
    if (pdm_dev_handle) {
        int ret = hal_pdm_init(pdm_dev_handle);
        if (ret == VPI_SUCCESS) {
            ret = hal_pdm_start(pdm_dev_handle, &pdm_stream_config);
            if (ret != VPI_SUCCESS) {
                printf("PDM start failed!\n");
            }
        } else {
            printf("PDM init failed!\n");
        }
    } else {
        printf("PDM get device failed!\n");
    }

    while (1)
    {
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
