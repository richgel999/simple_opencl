// ocl_device.h
#pragma once
#include <stdlib.h>
#include <stdint.h>

bool opencl_init(bool force_serialization);
void opencl_deinit();
bool opencl_is_available();

struct opencl_context;

// Each thread calling OpenCL should have its own opencl_context_ptr. This corresponds to a OpenCL command queue. (Confusingly, we only use a single OpenCL device "context".)
typedef opencl_context* opencl_context_ptr;

opencl_context_ptr opencl_create_context();
void opencl_destroy_context(opencl_context_ptr context);

// Example thread-safe processing function.
bool opencl_process_buffer(opencl_context_ptr context, const uint8_t *pInput_buf, uint8_t *pOutput_buf, uint32_t buf_size);

