/*
 * constants.c
 *
 *  Created on: 2025年7月21日
 *      Author: 廖先强
 */


#include "constants.h"
#include <string.h>

void circularbuffer_init(CircularBuffer *cb, uint8_t *buf, uint32_t size)
{
    cb->buffer = buf;
    cb->size = size;
    cb->head = 0;
    cb->tail = 0;
    cb->count = 0;
}

uint32_t circularbuffer_write(CircularBuffer *cb, const uint8_t *data, uint32_t len)
{
    uint32_t bytes_to_write = len;
    if (cb->count + len > cb->size) {
        bytes_to_write = cb->size - cb->count; // 只能写入剩余空间
    }

    if (bytes_to_write == 0) {
        return 0; // 缓冲区已满或没有数据可写
    }

    // 计算从head到缓冲区末尾的连续空间
    uint32_t end_space = cb->size - cb->head;

    if (bytes_to_write <= end_space) {
        // 数据可以一次性写入
        memcpy(&cb->buffer[cb->head], data, bytes_to_write);
        cb->head = (cb->head + bytes_to_write) % cb->size;
    } else {
        // 数据需要分两次写入（绕过末尾）
        memcpy(&cb->buffer[cb->head], data, end_space);
        memcpy(&cb->buffer[0], data + end_space, bytes_to_write - end_space);
        cb->head = bytes_to_write - end_space;
    }
    cb->count += bytes_to_write;
    return bytes_to_write;
}

uint32_t circularbuffer_read(CircularBuffer *cb, uint8_t *data, uint32_t len)
{
    uint32_t bytes_to_read = len;
    if (bytes_to_read > cb->count) {
        bytes_to_read = cb->count; // 只能读取已有的数据
    }

    if (bytes_to_read == 0) {
        return 0; // 缓冲区为空或没有数据可读
    }

    // 计算从tail到缓冲区末尾的连续数据
    uint32_t end_data = cb->size - cb->tail;

    if (bytes_to_read <= end_data) {
        // 数据可以一次性读取
        memcpy(data, &cb->buffer[cb->tail], bytes_to_read);
        cb->tail = (cb->tail + bytes_to_read) % cb->size;
    } else {
        // 数据需要分两次读取（绕过末尾）
        memcpy(data, &cb->buffer[cb->tail], end_data);
        memcpy(data + end_data, &cb->buffer[0], bytes_to_read - end_data);
        cb->tail = bytes_to_read - end_data;
    }
    cb->count -= bytes_to_read;
    return bytes_to_read;
}

uint32_t circularbuffer_getusedsize(CircularBuffer *cb)
{
    return cb->count;
}

