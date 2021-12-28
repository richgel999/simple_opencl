// File: ocl_device.cpp
#include "ocl_device.h"

// If this is 1, the OpenCL kernel program source will come from the char array in ocl_kernels.h. The advantage to this method is no external file I/O access.
// Otherwise, it will attempt to load the kernel program source from disk.
#define OCL_USE_KERNELS_HEADER (1)

#if OCL_USE_KERNELS_HEADER
#include "ocl_kernels.h"
#endif

#define OPENCL_ASSERT_ON_ANY_ERRORS (1)
#include "simple_ocl_wrapper.h"

// If 1, the kernel source code will come from encoders/ocl_kernels.h. Otherwise, it will be read from the "ocl_kernels.cl" file in the current directory (for development).
#define OCL_KERNELS_FILENAME "ocl_kernels.cl"

// Library global state
ocl g_ocl;

// All per-thread state goes here
struct opencl_context
{
	uint32_t m_ocl_total_pixel_blocks;
	cl_mem m_ocl_pixel_blocks;

	cl_command_queue m_command_queue;

	cl_kernel m_ocl_process_buffer_kernel;
};

static bool read_file_to_vec(const char* pFilename, std::vector<uint8_t>& data)
{
	FILE* pFile = nullptr;
#ifdef _WIN32
	fopen_s(&pFile, pFilename, "rb");
#else
	pFile = fopen(pFilename, "rb");
#endif
	if (!pFile)
		return false;
			
	fseek(pFile, 0, SEEK_END);
#ifdef _WIN32
	int64_t filesize = _ftelli64(pFile);
#else
	int64_t filesize = ftello(pFile);
#endif
	if (filesize < 0)
	{
		fclose(pFile);
		return false;
	}
	fseek(pFile, 0, SEEK_SET);

	if (sizeof(size_t) == sizeof(uint32_t))
	{
		if (filesize > 0x70000000)
		{
			// File might be too big to load safely in one alloc
			fclose(pFile);
			return false;
		}
	}

	data.resize((size_t)filesize);

	if (filesize)
	{
		if (fread(&data[0], 1, (size_t)filesize, pFile) != (size_t)filesize)
		{
			fclose(pFile);
			return false;
		}
	}

	fclose(pFile);
	return true;
}
		
bool opencl_init(bool force_serialization)
{
	if (g_ocl.is_initialized())
	{
		assert(0);
		return false;
	}

	if (!g_ocl.init(force_serialization))
	{
		ocl_error_printf("opencl_init: Failed initializing OpenCL\n");
		return false;
	}

	const char* pKernel_src = nullptr;
	size_t kernel_src_size = 0;

#if OCL_USE_KERNELS_HEADER
	pKernel_src = (const char *)ocl_kernels_cl;
	kernel_src_size = ocl_kernels_cl_len;

	printf("Using kernel source code from array in header src/ocl_kernels.h\n");
#else
	std::vector<uint8_t> kernel_src;

	// Read the text file containing the OpenCL kernels into the buffer.
	// You could also embed the OpenCL kernel source into the app using the "xxd" Linux tool.
	if (!read_file_to_vec(OCL_KERNELS_FILENAME, kernel_src))
	{
		ocl_error_printf("opencl_init: Cannot read OpenCL kernel source file \"%s\"! Make sure the current directory is \"bin\".\n", OCL_KERNELS_FILENAME);
		g_ocl.deinit();
		return false;
	}

	printf("Read kernel source from file \"%s\"\n", OCL_KERNELS_FILENAME);
		
	pKernel_src = (char*)kernel_src.data();
	kernel_src_size = kernel_src.size();
#endif
	
	if (!kernel_src_size)
	{
		ocl_error_printf("opencl_init: Invalid OpenCL kernel source file \"%s\"\n", OCL_KERNELS_FILENAME);
		g_ocl.deinit();
		return false;
	}

	if (!g_ocl.init_program(pKernel_src, kernel_src_size))
	{
		ocl_error_printf("opencl_init: Failed compiling OpenCL program\n");
		g_ocl.deinit();
		return false;
	}
							
	printf("OpenCL context initialized successfully\n");

	return true;
}

void opencl_deinit()
{
	g_ocl.deinit();
}

bool opencl_is_available()
{
	return g_ocl.is_initialized();
}

opencl_context_ptr opencl_create_context()
{
	if (!opencl_is_available())
	{
		ocl_error_printf("opencl_create_context: OpenCL not initialized\n");
		assert(0);
		return nullptr;
	}

	opencl_context* pContext = static_cast<opencl_context * >(calloc(sizeof(opencl_context), 1));
	if (!pContext)
		return nullptr;
			
	// To avoid driver bugs in some drivers - serialize this. Likely not necessary, we don't know.
	// https://community.intel.com/t5/OpenCL-for-CPU/Bug-report-clCreateKernelsInProgram-is-not-thread-safe/td-p/1159771
	
	pContext->m_command_queue = g_ocl.create_command_queue();
	if (!pContext->m_command_queue)
	{
		ocl_error_printf("opencl_create_context: Failed creating OpenCL command queue!\n");
		opencl_destroy_context(pContext);
		return nullptr;
	}

	// Create our kernel(s) here.
	pContext->m_ocl_process_buffer_kernel = g_ocl.create_kernel("process_buffer");
	if (!pContext->m_ocl_process_buffer_kernel)
	{
		ocl_error_printf("opencl_create_context: Failed creating OpenCL kernel process_buffer\n");
		opencl_destroy_context(pContext);
		return nullptr;
	}

	return pContext;
}

void opencl_destroy_context(opencl_context_ptr pContext)
{
	if (!pContext)
		return;

	g_ocl.destroy_kernel(pContext->m_ocl_process_buffer_kernel);

	g_ocl.destroy_command_queue(pContext->m_command_queue);
		
	memset(pContext, 0, sizeof(opencl_context));

	free(pContext);
}

// Example thread-safe function to process a buffer and return some output.
bool opencl_process_buffer(
	opencl_context_ptr pContext,
	const uint8_t* pBuffer,
	uint8_t* pOutput_buffer,
	uint32_t buffer_size)
{
	if (!opencl_is_available())
		return false;

	bool status = false;

	// Create input/output OpenCL buffers.			
	cl_mem input_buf = g_ocl.alloc_and_init_read_buffer(pContext->m_command_queue, pBuffer, buffer_size);
	cl_mem output_buf = g_ocl.alloc_write_buffer(buffer_size);

	if (!input_buf || !output_buf)
		goto exit;

	// Set the kernel arguments
	if (!g_ocl.set_kernel_args(pContext->m_ocl_process_buffer_kernel, input_buf, output_buf, buffer_size))
		goto exit;

	// Run the kernel
	if (!g_ocl.run_2D(pContext->m_command_queue, pContext->m_ocl_process_buffer_kernel, buffer_size, 1))
		goto exit;

	// Retrieve the output
	if (!g_ocl.read_from_buffer(pContext->m_command_queue, output_buf, pOutput_buffer, buffer_size))
		goto exit;

	status = true;

exit:
	g_ocl.destroy_buffer(input_buf);
	g_ocl.destroy_buffer(output_buf);

	return status;
}
