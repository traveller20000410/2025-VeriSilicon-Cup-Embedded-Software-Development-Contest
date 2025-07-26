/*
 * Copyright (c) 2019 Nuclei Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 /* linker script of normal firmware */

OUTPUT_ARCH( "riscv" )

ENTRY( _start )

MEMORY
{
  flash : ORIGIN = 0x10010000 + 320K, LENGTH = 3072K - 320K
  sdram : ORIGIN = 0x20200000, LENGTH = 192K /* Non-retention */
  siram : ORIGIN = 0x20230000, LENGTH = 64K  /* Retention */
  ble   : ORIGIN = 0x20240000, LENGTH = 0x20D70
  ipc   : ORIGIN = 0x20260D70, LENGTH = 0x10E0
  ipciso   : ORIGIN = 0x20261E50, LENGTH = 0x21B0
  noncache : ORIGIN = 0x20264000, LENGTH = 48K
  ilm : ORIGIN = 0x80000000, LENGTH = 64K
  dlm : ORIGIN = 0x90000000, LENGTH = 64K
}
__FLASH_MEM_END = ORIGIN(flash) + LENGTH(flash);
__SDRAM_MEM_END = ORIGIN(sdram) + LENGTH(sdram);
__SIRAM_MEM_END = ORIGIN(siram) + LENGTH(siram);
__NONCACHE_MEM_END = ORIGIN(noncache) + LENGTH(noncache);
__ILM_MEM_END = ORIGIN(ilm) + LENGTH(ilm);
__DLM_MEM_END = ORIGIN(dlm) + LENGTH(dlm);

PROVIDE(__STACK_SIZE = 2K);
PROVIDE(__HEAP_SIZE = 14K);
__TOT_STACK_SIZE = __STACK_SIZE;

__FLASH_PAGE_SIZE = 256;
__XIP_BASE_ADDR = 0x10000000;
__FW_FLASH_BASE = ORIGIN(flash) - __XIP_BASE_ADDR;
__FW_XIP_BASE = ORIGIN(flash);

__IPC_MEM_SIZE = LENGTH(ipc);
PROVIDE(__zsp_addr = 0x10010000 - __XIP_BASE_ADDR);

SECTIONS
{
    /******************************ilm********************************************/
    __ILM_VMA_BASE = ORIGIN(ilm);
    __ILM_LMA_BASE = ORIGIN(flash) + __FLASH_PAGE_SIZE;
    .ilm __ILM_VMA_BASE :
    {
        PROVIDE(__fw_entry = .);

        *(.vtable)
        KEEP (*(SORT_NONE(.init)))
        *(.text.unlikely .text.unlikely.*)
        *(.text.startup .text.startup.*)
        . = ALIGN(4);

/* notice: put them in call order for cache hint rate */
        *system_pegasus*.o(.text*)
        *soc_init*.o(.text*)
        *soc_sysctl*.o(.text*)
        *soc_pinmux*.o(.text*)
        *soc_clk*.o(.text*)
        *hal_clock*.o(.text*)
        *clock*.o(.text*)
        *clk_*.o(.text*)
        *vs_iot_pll_*.o(.text*)
        *vs_iot_fix*.o(.text*)
        *hal_flash*.o(.text*)
        *hal_spi*.o(.text*)
        *vs_spiflash*.o(.text*)
        *vs_qspi*.o(.text*)
        *vsd_common*.o(.text*)
        *(.nonxip_text)
/* notice: all above codes are for environment setup */

/* notice: put OS in ITCM for performance and mutex is used by Flash */
        *portasm*.o(.text*)
        *port*.o(.text*)
        *libos*.a:*(.text*)
        *os/FreeRTOS/Kernel/tasks.o(.text*)
        *os/FreeRTOS/Kernel/queue.o(.text*)
        *os/FreeRTOS/Kernel/list.o(.text*)
        *os/FreeRTOS/Kernel/timers.o(.text*)
        *os/FreeRTOS/Kernel/stream_buffer.o(.text*)
        *os/FreeRTOS/Kernel/croutine.o(.text*)
        *os/FreeRTOS/Kernel/event_groups.o(.text*)
        *os/FreeRTOS/Kernel/portable/Common/mpu_wrappers.o(.text*)
        *os/FreeRTOS/Kernel/portable/MemMang/heap_4.o(.text*)
        *osal_task*.o(.text*)
        *osal_event*.o(.text*)
        *osal_heap*.o(.text*)
        *osal_lock*.o(.text*)
        *osal_notify*.o(.text*)
        *osal_time*.o(.text*)
        *osal_semaphore*.o(.text*)
        *osal_sys_state*.o(.text*)
        *heap_4_noncache*.o(.text*)
        *soc_irq*.o(.text*)
/* cyw add */      
        *vad.o(.text*)
        *conv.o(.text*)
        *algo_task.o(.text*)
        *speech_task.o(.text*)
        *(.ilm_section) 
        
        *(.itext)
        . = ALIGN(64);
    }
    __ILM_SECTION_SIZE = SIZEOF(.ilm);
    __ILM_SECTION_PAGES = __ILM_SECTION_SIZE / __FLASH_PAGE_SIZE;
    __ILM_SECTION_PAGES += (((__ILM_SECTION_SIZE % __FLASH_PAGE_SIZE) > 0 ) ? 1 : 0);
    __ILM_SECTION_PAGES += 1;
    __ILM_VMA_END = __ILM_VMA_BASE + __FLASH_PAGE_SIZE * __ILM_SECTION_PAGES;
    __ILM_LMA_END = __ILM_LMA_BASE + __FLASH_PAGE_SIZE * __ILM_SECTION_PAGES;

    ASSERT((__ILM_SECTION_SIZE) <= LENGTH(ilm), "ilm space error!")
    /****************************************************************************/

    /************************************siram**********************************/
    __SIRAM_VMA_BASE = ORIGIN(siram);
    __SIRAM_LMA_BASE = __ILM_LMA_END + __FLASH_PAGE_SIZE;
    .siram __SIRAM_VMA_BASE :
    {
/* notice: for fastboot */
        *(.nonxip_text)
        *gpd*.o(.text*)
        *pmu*.o(.text*)
        *pm*.o(.text*)
        *rtc*.o(.text*)
        *notifier*.o(.text*)
        *libc_nano.a:*(.text*)
        *libg_nano.a:*(.text*)
        *libgcc.a:*(.text*)
        *newlib*.o(.text*)
        *libnosys.a:*(.text*)

        . = ALIGN(8);
        *(.gnu.linkonce.d.*)

/* notice: for XiP */
        . = ALIGN(8);
        *system_pegasus*.o(.rodata* .srodata*)
        *soc_init*.o(.rodata* .srodata*)
        *soc_pinmux*.o(.rodata* .srodata*)
        *soc_clk*.o(.rodata* .srodata*)
        *hal_clock*.o(.rodata* .srodata*)
        *clk_*.o(.rodata* .srodata*)
        *vs_iot_fix*.o(.rodata* .srodata*)
        *vs_iot_pll_*.o(.rodata* .srodata*)
        *vs_spiflash*.o(.rodata* .srodata*)
        *vs_qspi*.o(.rodata* .srodata*)
        *port*.o(.rodata* .srodata*)
        *libos*.a:*(.rodata* .srodata*)
        *libc_nano.a:*(.rodata* .srodata*)
        *libg_nano.a:*(.rodata* .srodata*)
        *libgcc.a:*(.rodata* .srodata*)
        *soc_irq*.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/tasks.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/queue.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/list.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/timers.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/stream_buffer.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/croutine.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/event_groups.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/portable/Common/mpu_wrappers.o(.rodata* .srodata*)
        *os/FreeRTOS/Kernel/portable/MemMang/heap_4.o(.rodata* .srodata*)
        *osal_task*.o(.rodata* .srodata*)
        *osal_event*.o(.rodata* .srodata*)
        *osal_heap*.o(.rodata* .srodata*)
        *osal_lock*.o(.rodata* .srodata*)
        *osal_notify*.o(.rodata* .srodata*)
        *osal_time*.o(.rodata* .srodata*)
        *osal_semaphore*.o(.rodata* .srodata*)
        *osal_sys_state*.o(.rodata* .srodata*)


/* notice: for fastboot */
        *gpd*.o(.rodata* .srodata*)
        *pmu*.o(.rodata* .srodata*)
        *pm*.o(.rodata* .srodata*)
        *rtc*.o(.rodata* .srodata*)
        *notifier*.o(.rodata* .srodata*)
        . = ALIGN(64);
    }
    __SIRAM_SECTION_SIZE = SIZEOF(.siram);
    __SIRAM_SECTION_PAGES = __SIRAM_SECTION_SIZE / __FLASH_PAGE_SIZE;
    __SIRAM_SECTION_PAGES += (((__SIRAM_SECTION_SIZE % __FLASH_PAGE_SIZE) > 0 ) ? 1 : 0);
    __SIRAM_SECTION_PAGES += 1;
    __SIRAM_VMA_END = __SIRAM_VMA_BASE + __SIRAM_SECTION_PAGES * __FLASH_PAGE_SIZE;
    __SIRAM_LMA_END = __SIRAM_LMA_BASE +  __SIRAM_SECTION_PAGES * __FLASH_PAGE_SIZE;

    ASSERT((__SIRAM_SECTION_SIZE) <= LENGTH(siram), "siram space error!")
    /**************************************************************************/

    /***********************************sdram**********************************/
    __SDRAM_VMA_BASE = ORIGIN(sdram);
    __SDRAM_LMA_BASE = __SIRAM_LMA_END + __FLASH_PAGE_SIZE;
    .sdram __SDRAM_VMA_BASE : ALIGN(8)
    {
        . = ALIGN(4);
        _net_buf_pool_list = .;
        _net_buf_pool_list_start = .;
        *(._net_buf_pool.*);
        _net_buf_pool_list_end = .;

        . = ALIGN(4);
        _k_queue_list_start  = .;
        *(._k_queue.static.*);
        _k_queue_list_end  = .;

        . = ALIGN(4);
        _k_mem_slab_list_start  = .;
        KEEP(*(._k_mem_slab.static.*));
        _k_mem_slab_list_end  = .;

        . = ALIGN(4);
        _bt_gatt_service_static_list_start  = .;
        KEEP(*(._bt_gatt_service_static.static.*));
        _bt_gatt_service_static_list_end  = .;

        . = ALIGN(4);
        _bt_l2cap_fixed_chan_list_start  = .;
        KEEP(*(._bt_l2cap_fixed_chan.static.*));
        _bt_l2cap_fixed_chan_list_end  = .;

        . = ALIGN(4);
        _settings_handler_static_list_start  = .;
        KEEP(*(._settings_handler_static.static.*));
        _settings_handler_static_list_end  = .;

        . = ALIGN(4);
        _k_sem_list_start  = .;
        KEEP(*(._k_sem.static.*));
        _k_sem_list_end  = .;

        . = ALIGN(4);
        _k_mutex_list_start  = .;
        KEEP(*(._k_mutex.static.*));
        _k_mutex_list_end  = .;

        . = ALIGN(4);
        _bt_conn_cb_list_start  = .;
        KEEP(*(._bt_conn_cb.static.*));
        _bt_conn_cb_list_end  = .;

        . = ALIGN(8);
        PROVIDE( __global_pointer$ = . + 0x800 );
        *(.sdata .sdata.* .sdata*)
        *(.data*)
        *(.sdram_data)
        *(.sys_inherited_data)
        *(.gnu.linkonce.s.*)
        . = ALIGN(8);

        PROVIDE( __tls_base = . );
        *(.tdata .tdata.* .gnu.linkonce.td.*)
        . = ALIGN(64);
    }
    __SDRAM_SECTION_SIZE = SIZEOF(.sdram);
    __SDRAM_SECTION_PAGES = __SDRAM_SECTION_SIZE / __FLASH_PAGE_SIZE;
    __SDRAM_SECTION_PAGES += (((__SDRAM_SECTION_SIZE % __FLASH_PAGE_SIZE) > 0 ) ? 1 : 0);
    __SDRAM_SECTION_PAGES += 1;
    __SDRAM_VMA_END = __SDRAM_VMA_BASE + __SDRAM_SECTION_SIZE;
    __SDRAM_LMA_END = __SDRAM_LMA_BASE + __SDRAM_SECTION_PAGES * __FLASH_PAGE_SIZE;

    PROVIDE( _data_lma = LOADADDR(.sdram) );
    PROVIDE( _data = ADDR(.sdram) );
    PROVIDE( _edata = . );
    /*****************************************************************************/

    /*********************************flash***************************************/
    __FLASH_VMA_BASE = __SDRAM_LMA_END + __FLASH_PAGE_SIZE;
    __FLASH_LMA_BASE = __SDRAM_LMA_END + __FLASH_PAGE_SIZE;
    .flash __FLASH_VMA_BASE :
    {
        . = ALIGN(4);
        *libble*.a:*(.text*)
        . = ALIGN(4);
        __init_start = .;
        __init_PRE_KERNEL_1_start = .;
        KEEP(*(SORT(.z_init_PRE_KERNEL_1[0-9]_*)));
        KEEP(*(SORT(.z_init_PRE_KERNEL_1[1-9][0-9]_*)));
        __init_PRE_KERNEL_2_start = .;
        KEEP(*(SORT(.z_init_PRE_KERNEL_2[0-9]_*)));
        KEEP(*(SORT(.z_init_PRE_KERNEL_2[1-9][0-9]_*)));
        __init_POST_KERNEL_start = .;
        KEEP(*(SORT(.z_init_POST_KERNEL[0-9]_*)));
        KEEP(*(SORT(.z_init_POST_KERNEL[1-9][0-9]_*)));
        __init_APPLICATION_start = .;
        KEEP(*(SORT(.z_init_APPLICATION[0-9]_*)));
        KEEP(*(SORT(.z_init_APPLICATION[1-9][0-9]_*)));
        __init_end = .;

        . = ALIGN(4);
        *(.stext)
        . = ALIGN(4);
        *(.text*)

        . = ALIGN(8);
        *(.srodata.cst*)
        *(.srodata .srodata.*)
        *(.rdata)
        *(.rodata .rodata.*)
        *(.gnu.linkonce.r.*)
        . = ALIGN(64 * 1024);
    }
    __FLASH_SECTION_SIZE = SIZEOF(.flash);
    __FLASH_SECTION_PAGES = __FLASH_SECTION_SIZE / __FLASH_PAGE_SIZE;
    __FLASH_SECTION_PAGES += (((__FLASH_SECTION_SIZE % __FLASH_PAGE_SIZE) > 0 ) ? 1 : 0);
    __FLASH_SECTION_PAGES += 1;
    __FLASH_VMA_END = __FLASH_VMA_BASE + __FLASH_SECTION_PAGES * __FLASH_PAGE_SIZE;
    __FLASH_LMA_END = __FLASH_LMA_BASE + __FLASH_SECTION_PAGES * __FLASH_PAGE_SIZE;

    PROVIDE( _text_lma = LOADADDR(.flash) );
    PROVIDE( _text = ADDR(.flash) );
    PROVIDE (_etext = .);
    /***************************************************************************/

    /*****************************noinit and bss*******************************************/
    __NOINIT_VMA_BASE = __SDRAM_VMA_END;
    .noinit __NOINIT_VMA_BASE (NOLOAD): ALIGN(8)
    {
        *(.noinit.*)
        *(.sensor_data)
        *(.rsv_data)
        . = ALIGN(4);
    }
    __NOINIT_SECTION_SIZE = SIZEOF(.noinit);
    __NOINIT_VMA_END = __NOINIT_VMA_BASE + __NOINIT_SECTION_SIZE;

    __BSS_VMA_BASE = __NOINIT_VMA_END;
    PROVIDE( _fbss = __BSS_VMA_BASE);
    PROVIDE( __bss_start = __BSS_VMA_BASE);
    .bss __BSS_VMA_BASE (NOLOAD): ALIGN(4)
    {
       *(.sbss*)
       *(.gnu.linkonce.sb.*)
       *(.bss .bss.*)
       *(.gnu.linkonce.b.*)
       *(COMMON)
       . = ALIGN(4);
    }
    __BSS_SECTION_SIZE = SIZEOF(.bss);
    __BSS_VMA_END = __BSS_VMA_BASE + __BSS_SECTION_SIZE;
    PROVIDE(_end = .);

    ASSERT((__BSS_VMA_END - __SDRAM_VMA_BASE) <= (LENGTH(sdram)), "sdram space error!")
    /****************************************************************************/

    /**************************heap and stack***********************/
    __NON_CACHEABLE_VMA_BASE = ORIGIN(noncache);
    .noncacheable __NON_CACHEABLE_VMA_BASE (NOLOAD) :
    {
        *(.noncache_data)
    }
    __NON_CACHEABLE_SIZE = SIZEOF(.noncacheable);
    __NON_CACHEABLE_VMA_END =  __NON_CACHEABLE_VMA_BASE + __NON_CACHEABLE_SIZE;

    __HEAP_VMA_BASE = __NONCACHE_MEM_END - (__HEAP_SIZE + __TOT_STACK_SIZE);
    .heap __HEAP_VMA_BASE (NOLOAD) : ALIGN(16)
    {
        . = ALIGN(16);
        PROVIDE( __heap_start = . );
        PROVIDE(end = .);
        . += __HEAP_SIZE;
        . = ALIGN(16);
        PROVIDE( __heap_limit = . );
    }
    __HEAP_SECTION_SIZE = SIZEOF(.heap);
    __HEAP_VMA_END = __HEAP_VMA_BASE + __HEAP_SECTION_SIZE;

    __STACK_VMA_BASE = __HEAP_VMA_END;
    .stack __STACK_VMA_BASE (NOLOAD) :
    {
        . = ALIGN(16);
        PROVIDE( _heap_end = . );
        PROVIDE( __heap_end = . );
        PROVIDE( __StackLimit = . );
        PROVIDE( __StackBottom = . );
        . += __TOT_STACK_SIZE;
        . = ALIGN(16);
        PROVIDE( __StackTop = (. - 16));
        PROVIDE( _sp = (. - 16));
    }
    __STACK_SECTION_SIZE = SIZEOF(.stack);
    __STACK_VMA_END = __STACK_VMA_BASE + __STACK_SECTION_SIZE;

    ASSERT((__STACK_VMA_END - __NON_CACHEABLE_VMA_BASE) <= LENGTH(noncache), "Non cacheable space error!")
    /****************************************************************************/


/******************************** DLM 静态缓冲区 ************************************/
__DLM_BSS_VMA_BASE = ORIGIN(dlm);
.dlm_bss __DLM_BSS_VMA_BASE (NOLOAD) : ALIGN(8)
{
    /* 放置所有使用 __attribute__((section(".dlm_bss"))) 的静态数组 */
    *(.dlm_bss)
    . = ALIGN(8);
}
__DLM_BSS_SECTION_SIZE = SIZEOF(.dlm_bss);
__DLM_BSS_VMA_END = __DLM_BSS_VMA_BASE + __DLM_BSS_SECTION_SIZE;
/********************************************************************/


    /********************************os heap************************************/
    __OSHEAP_VMA_BASE = ORIGIN(dlm);
    .osheap __OSHEAP_VMA_BASE (NOLOAD) : ALIGN(8)
    {
        *(.RTOS_HEAP_SEC)
    }
    __OSHEAP_SECTION_SIZE = SIZEOF(.osheap);
    __OSHEAP_VMA_END = __OSHEAP_VMA_BASE + __OSHEAP_SECTION_SIZE;

    ASSERT((__OSHEAP_SECTION_SIZE) <= LENGTH(dlm), "dlm space error!")
    /**************************************************************************/

    /********************************ipc************************************/
    __IPC_VMA_BASE = ORIGIN(ipc);
    .ipc __IPC_VMA_BASE (NOLOAD) :
    {
        __ipc_mem_start = .;
    }
    /**************************************************************************/
}
