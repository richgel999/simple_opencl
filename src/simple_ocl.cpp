// simple_ocl.cpp
// Simple OpenCL example 
#include "ocl_device.h"
#include <stdio.h>
#include <vector>

int main(int arg_c, char **arg_v)
{
	// Create the OpenCL device.
	if (!opencl_init(false))
	{
		fprintf(stderr, "Failed initializing OpenCL!\n");
		return EXIT_FAILURE;
	}

	// Create our thread-local OpenCL context. Each thread will need its own context.
	opencl_context_ptr pContext = opencl_create_context();
	if (!pContext)
	{
		fprintf(stderr, "Failed creating OpenCL context!\n");
		
		opencl_deinit();

		return EXIT_FAILURE;
	}

	// Now create some data to process, and an output buffer.
	printf("Running \"process_buffer\" kernel\n");
	
	const uint32_t BUF_SIZE = 8192;
	std::vector<uint8_t> in_buf(BUF_SIZE);
	for (uint32_t i = 0; i < BUF_SIZE; i++)
		in_buf[i] = (uint8_t)rand();

	std::vector<uint8_t> out_buf(BUF_SIZE);

	// Invoke the kernel.
	if (!opencl_process_buffer(pContext, in_buf.data(), out_buf.data(), BUF_SIZE))
	{
		printf("Failed running OpenCL kernel!\n");
		
		opencl_destroy_context(pContext);
		opencl_deinit();

		return EXIT_FAILURE;
	}

	// Check the output buffer for correctness
	uint32_t total_failures = 0;
	for (uint32_t i = 0; i < BUF_SIZE; i++)
	{
		if (out_buf[i] != (in_buf[i] ^ (uint8_t)i))
		{
			printf("Validation failed at offset %u\n", i);
			total_failures++;
		}
	}
	
	if (!total_failures)
	{
		printf("Validation succeeded\n");
	}

	printf("Input/output buffer contents (first 16 bytes):\n");
	for (uint32_t i = 0; i < 16; i++)
	{
		printf("%u %u\n", in_buf[i], out_buf[i]);
	}

	// Destroy the context and device.
	opencl_destroy_context(pContext);
	opencl_deinit();

	return total_failures ? EXIT_FAILURE : EXIT_SUCCESS;
}

