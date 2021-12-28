// simple_ocl_wrapper.h
// Our simple C++ wrapper. You can also use OpenCL's built-in wrapper classes, but I like the C API better and this way I know exactly what's happening.

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <mutex>
#include <assert.h>
#include <stdarg.h>

// We only use OpenCL v1.2 or less.
#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

inline void ocl_error_printf(const char* pFmt, ...)
{
	va_list args;
	va_start(args, pFmt);
	vfprintf(stderr, pFmt, args);
	va_end(args);

#if OPENCL_ASSERT_ON_ANY_ERRORS
	assert(0);
#endif
}
   	
class ocl
{
public:
	ocl() 
	{
		memset(&m_dev_fp_config, 0, sizeof(m_dev_fp_config));
		
		m_ocl_mutex.lock();
		m_ocl_mutex.unlock();
	}

	~ocl()
	{
	}

	bool is_initialized() const { return m_device_id != nullptr; }

	cl_device_id get_device_id() const { return m_device_id; }
	cl_context get_context() const { return m_context; }
	cl_command_queue get_command_queue() { return m_command_queue; }
	cl_program get_program() const { return m_program; }

	bool init(bool force_serialization)
	{
		deinit();

		cl_uint num_platforms = 0;
		cl_int ret = clGetPlatformIDs(0, NULL, &num_platforms);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::init: clGetPlatformIDs() failed with %i\n", ret);
			return false;
		}

		if ((!num_platforms) || (num_platforms > INT_MAX))
		{
			ocl_error_printf("ocl::init: clGetPlatformIDs() returned an invalid number of num_platforms\n");
			return false;
		}

		std::vector<cl_platform_id> platforms(num_platforms);

		ret = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::init: clGetPlatformIDs() failed\n");
			return false;
		}

		cl_uint num_devices = 0;
		ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &m_device_id, &num_devices);

		if (ret == CL_DEVICE_NOT_FOUND)
		{
			ocl_error_printf("ocl::init: Couldn't get any GPU device ID's, trying CL_DEVICE_TYPE_CPU\n");

			ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &m_device_id, &num_devices);
		}

		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::init: Unable to get any device ID's\n");

			m_device_id = nullptr;
			return false;
		}

		ret = clGetDeviceInfo(m_device_id,
			CL_DEVICE_SINGLE_FP_CONFIG,
			sizeof(m_dev_fp_config),
			&m_dev_fp_config,
			nullptr);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::init: clGetDeviceInfo() failed\n");
			return false;
		}

		char plat_vers[256] = { 0 };

		size_t rv = 0;
		ret = clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, sizeof(plat_vers), plat_vers, &rv);
		if (ret == CL_SUCCESS)
			printf("OpenCL platform version: \"%s\"\n", plat_vers);

		// Serialize CL calls with the AMD driver to avoid lockups when multiple command queues per thread are used. This sucks, but what can we do?
		m_use_mutex = (strstr(plat_vers, "AMD") != nullptr) || force_serialization;

		printf("Serializing OpenCL calls across threads: %u\n", (uint32_t)m_use_mutex);

		m_context = clCreateContext(nullptr, 1, &m_device_id, nullptr, nullptr, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::init: clCreateContext() failed\n");

			m_device_id = nullptr;
			m_context = nullptr;
			return false;
		}

		m_command_queue = clCreateCommandQueue(m_context, m_device_id, 0, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::init: clCreateCommandQueue() failed\n");

			deinit();
			return false;
		}
					
		printf("OpenCL device initialized successfully\n");

		return true;
	}
			
	bool deinit()
	{
		if (m_program)
		{
			clReleaseProgram(m_program);
			m_program = nullptr;
		}

		if (m_command_queue)
		{
			clReleaseCommandQueue(m_command_queue);
			m_command_queue = nullptr;
		}

		if (m_context)
		{
			clReleaseContext(m_context);
			m_context = nullptr;
		}

		m_device_id = nullptr;

		return true;
	}

	cl_command_queue create_command_queue()
	{
		cl_serializer serializer(this);

		cl_int ret = 0;
		cl_command_queue p = clCreateCommandQueue(m_context, m_device_id, 0, &ret);
		if (ret != CL_SUCCESS)
			return nullptr;

		return p;
	}

	void destroy_command_queue(cl_command_queue p)
	{
		if (p)
		{
			cl_serializer serializer(this);

			clReleaseCommandQueue(p);
		}
	}

	bool init_program(const char* pSrc, size_t src_size)
	{
		cl_int ret;

		if (m_program != nullptr)
		{
			clReleaseProgram(m_program);
			m_program = nullptr;
		}

		m_program = clCreateProgramWithSource(m_context, 1, (const char**)&pSrc, (const size_t*)&src_size, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::init_program: clCreateProgramWithSource() failed!\n");
			return false;
		}

		std::string options;
		if (m_dev_fp_config & CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT)
		{
			options += "-cl-fp32-correctly-rounded-divide-sqrt";
		}

		options += " -cl-std=CL1.2";
		//options += " -cl-opt-disable";
		//options += " -cl-mad-enable";
		//options += " -cl-fast-relaxed-math";

		ret = clBuildProgram(m_program, 1, &m_device_id,
			options.size() ? options.c_str() : nullptr,  // options
			nullptr,  // notify
			nullptr); // user_data

		if (ret != CL_SUCCESS)
		{
			const cl_int build_program_result = ret;

			size_t ret_val_size;
			ret = clGetProgramBuildInfo(m_program, m_device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
			if (ret != CL_SUCCESS)
			{
				ocl_error_printf("ocl::init_program: clGetProgramBuildInfo() failed!\n");
				return false;
			}

			std::vector<char> build_log(ret_val_size + 1);

			ret = clGetProgramBuildInfo(m_program, m_device_id, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log.data(), NULL);

			ocl_error_printf("\nclBuildProgram() failed with error %i:\n%s", build_program_result, build_log.data());

			return false;
		}

		return true;
	}

	cl_kernel create_kernel(const char* pName)
	{
		if (!m_program)
			return nullptr;

		cl_serializer serializer(this);

		cl_int ret;
		cl_kernel kernel = clCreateKernel(m_program, pName, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::create_kernel: clCreateKernel() failed!\n");
			return nullptr;
		}

		return kernel;
	}

	bool destroy_kernel(cl_kernel k)
	{
		if (k)
		{
			cl_serializer serializer(this);

			cl_int ret = clReleaseKernel(k);
			if (ret != CL_SUCCESS)
			{
				ocl_error_printf("ocl::destroy_kernel: clReleaseKernel() failed!\n");
				return false;
			}
		}
		return true;
	}

	cl_mem alloc_read_buffer(size_t size)
	{
		cl_serializer serializer(this);

		cl_int ret;
		cl_mem obj = clCreateBuffer(m_context, CL_MEM_READ_ONLY, size, NULL, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::alloc_read_buffer: clCreateBuffer() failed!\n");
			return nullptr;
		}

		return obj;
	}

	cl_mem alloc_and_init_read_buffer(cl_command_queue command_queue, const void *pInit, size_t size)
	{
		cl_serializer serializer(this);

		cl_int ret;
		cl_mem obj = clCreateBuffer(m_context, CL_MEM_READ_ONLY, size, NULL, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::alloc_and_init_read_buffer: clCreateBuffer() failed!\n");
			return nullptr;
		}

		ret = clEnqueueWriteBuffer(command_queue, obj, CL_TRUE, 0, size, pInit, 0, NULL, NULL);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::alloc_and_init_read_buffer: clEnqueueWriteBuffer() failed!\n");
			return false;
		}

		return obj;
	}

	cl_mem alloc_write_buffer(size_t size)
	{
		cl_serializer serializer(this);

		cl_int ret;
		cl_mem obj = clCreateBuffer(m_context, CL_MEM_WRITE_ONLY, size, NULL, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::alloc_write_buffer: clCreateBuffer() failed!\n");
			return nullptr;
		}

		return obj;
	}
			
	bool destroy_buffer(cl_mem buf)
	{
		if (buf)
		{
			cl_serializer serializer(this);

			cl_int ret = clReleaseMemObject(buf);
			if (ret != CL_SUCCESS)
			{
				ocl_error_printf("ocl::destroy_buffer: clReleaseMemObject() failed!\n");
				return false;
			}
		}

		return true;
	}

	bool write_to_buffer(cl_command_queue command_queue, cl_mem clmem, const void* d, const size_t m)
	{
		cl_serializer serializer(this);

		cl_int ret = clEnqueueWriteBuffer(command_queue, clmem, CL_TRUE, 0, m, d, 0, NULL, NULL);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::write_to_buffer: clEnqueueWriteBuffer() failed!\n");
			return false;
		}

		return true;
	}

	bool read_from_buffer(cl_command_queue command_queue, const cl_mem clmem, void* d, size_t m)
	{
		cl_serializer serializer(this);

		cl_int ret = clEnqueueReadBuffer(command_queue, clmem, CL_TRUE, 0, m, d, 0, NULL, NULL);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::read_from_buffer: clEnqueueReadBuffer() failed!\n");
			return false;
		}

		return true;
	}

	cl_mem create_read_image_u8(uint32_t width, uint32_t height, const void* pPixels, uint32_t bytes_per_pixel, bool normalized)
	{
		cl_image_format fmt = get_image_format(bytes_per_pixel, normalized);

		cl_image_desc desc;
		memset(&desc, 0, sizeof(desc));
		desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		desc.image_width = width;
		desc.image_height = height;
		desc.image_row_pitch = width * bytes_per_pixel;

		cl_serializer serializer(this);

		cl_int ret;
		cl_mem img = clCreateImage(m_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &fmt, &desc, (void*)pPixels, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::create_read_image_u8: clCreateImage() failed!\n");
			return nullptr;
		}

		return img;
	}

	cl_mem create_write_image_u8(uint32_t width, uint32_t height, uint32_t bytes_per_pixel, bool normalized)
	{
		cl_image_format fmt = get_image_format(bytes_per_pixel, normalized);

		cl_image_desc desc;
		memset(&desc, 0, sizeof(desc));
		desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		desc.image_width = width;
		desc.image_height = height;

		cl_serializer serializer(this);

		cl_int ret;
		cl_mem img = clCreateImage(m_context, CL_MEM_WRITE_ONLY, &fmt, &desc, nullptr, &ret);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::create_write_image_u8: clCreateImage() failed!\n");
			return nullptr;
		}

		return img;
	}

	bool read_from_image(cl_command_queue command_queue, cl_mem img, void* pPixels, uint32_t ofs_x, uint32_t ofs_y, uint32_t width, uint32_t height)
	{
		cl_serializer serializer(this);

		size_t origin[3] = { ofs_x, ofs_y, 0 }, region[3] = { width, height, 1 };

		cl_int err = clEnqueueReadImage(command_queue, img, CL_TRUE, origin, region, 0, 0, pPixels, 0, NULL, NULL);
		if (err != CL_SUCCESS)
		{
			ocl_error_printf("ocl::read_from_image: clEnqueueReadImage() failed!\n");
			return false;
		}

		return true;
	}

	bool run_1D(cl_command_queue command_queue, const cl_kernel kernel, size_t num_items)
	{
		cl_serializer serializer(this);

		cl_int ret = clEnqueueNDRangeKernel(command_queue, kernel,
			1,  // work_dim
			nullptr, // global_work_offset
			&num_items, // global_work_size
			nullptr, // local_work_size
			0, // num_events_in_wait_list
			nullptr, // event_wait_list
			nullptr // event
		);

		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::run_1D: clEnqueueNDRangeKernel() failed!\n");
			return false;
		}

		return true;
	}

	bool run_2D(cl_command_queue command_queue, const cl_kernel kernel, size_t width, size_t height)
	{
		cl_serializer serializer(this);

		size_t num_global_items[2] = { width, height };
		//size_t num_local_items[2] = { 1, 1 };

		cl_int ret = clEnqueueNDRangeKernel(command_queue, kernel,
			2,  // work_dim
			nullptr, // global_work_offset
			num_global_items, // global_work_size
			nullptr, // local_work_size
			0, // num_events_in_wait_list
			nullptr, // event_wait_list
			nullptr // event
		);

		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::run_2D: clEnqueueNDRangeKernel() failed!\n");
			return false;
		}

		return true;
	}

	bool run_2D(cl_command_queue command_queue, const cl_kernel kernel, size_t ofs_x, size_t ofs_y, size_t width, size_t height)
	{
		cl_serializer serializer(this);

		size_t global_ofs[2] = { ofs_x, ofs_y };
		size_t num_global_items[2] = { width, height };
		//size_t num_local_items[2] = { 1, 1 };

		cl_int ret = clEnqueueNDRangeKernel(command_queue, kernel,
			2,  // work_dim
			global_ofs, // global_work_offset
			num_global_items, // global_work_size
			nullptr, // local_work_size
			0, // num_events_in_wait_list
			nullptr, // event_wait_list
			nullptr // event
		);

		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::run_2D: clEnqueueNDRangeKernel() failed!\n");
			return false;
		}

		return true;
	}

	void flush(cl_command_queue command_queue)
	{
		cl_serializer serializer(this);

		clFlush(command_queue);
		clFinish(command_queue);
	}

	template<typename T>
	bool set_kernel_arg(cl_kernel kernel, uint32_t index, const T& obj)
	{
		cl_serializer serializer(this);

		cl_int ret = clSetKernelArg(kernel, index, sizeof(T), (void*)&obj);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::set_kernel_arg: clSetKernelArg() failed!\n");
			return false;
		}
		return true;
	}

	template<typename T>
	bool set_kernel_args(cl_kernel kernel, const T& obj1)
	{
		cl_serializer serializer(this);

		cl_int ret = clSetKernelArg(kernel, 0, sizeof(T), (void*)&obj1);
		if (ret != CL_SUCCESS)
		{
			ocl_error_printf("ocl::set_kernel_arg: clSetKernelArg() failed!\n");
			return false;
		}
		return true;
	}

#define CHECK_ERR if (ret != CL_SUCCESS)	{ ocl_error_printf("ocl::set_kernel_args: clSetKernelArg() failed!\n"); return false; }

	template<typename T, typename U>
	bool set_kernel_args(cl_kernel kernel, const T& obj1, const U& obj2)
	{
		cl_serializer serializer(this);
		cl_int ret = clSetKernelArg(kernel, 0, sizeof(T), (void*)&obj1); CHECK_ERR
		ret = clSetKernelArg(kernel, 1, sizeof(U), (void*)&obj2); CHECK_ERR
		return true;
	}

	template<typename T, typename U, typename V>
	bool set_kernel_args(cl_kernel kernel, const T& obj1, const U& obj2, const V& obj3)
	{
		cl_serializer serializer(this);
		cl_int ret = clSetKernelArg(kernel, 0, sizeof(T), (void*)&obj1); CHECK_ERR
		ret = clSetKernelArg(kernel, 1, sizeof(U), (void*)&obj2); CHECK_ERR
		ret = clSetKernelArg(kernel, 2, sizeof(V), (void*)&obj3); CHECK_ERR
		return true;
	}

	template<typename T, typename U, typename V, typename W>
	bool set_kernel_args(cl_kernel kernel, const T& obj1, const U& obj2, const V& obj3, const W& obj4)
	{
		cl_serializer serializer(this);
		cl_int ret = clSetKernelArg(kernel, 0, sizeof(T), (void*)&obj1); CHECK_ERR
		ret = clSetKernelArg(kernel, 1, sizeof(U), (void*)&obj2); CHECK_ERR
		ret = clSetKernelArg(kernel, 2, sizeof(V), (void*)&obj3); CHECK_ERR
		ret = clSetKernelArg(kernel, 3, sizeof(W), (void*)&obj4); CHECK_ERR
		return true;
	}

	template<typename T, typename U, typename V, typename W, typename X>
	bool set_kernel_args(cl_kernel kernel, const T& obj1, const U& obj2, const V& obj3, const W& obj4, const X& obj5)
	{
		cl_serializer serializer(this);
		cl_int ret = clSetKernelArg(kernel, 0, sizeof(T), (void*)&obj1); CHECK_ERR
		ret = clSetKernelArg(kernel, 1, sizeof(U), (void*)&obj2); CHECK_ERR
		ret = clSetKernelArg(kernel, 2, sizeof(V), (void*)&obj3); CHECK_ERR
		ret = clSetKernelArg(kernel, 3, sizeof(W), (void*)&obj4); CHECK_ERR
		ret = clSetKernelArg(kernel, 4, sizeof(X), (void*)&obj5); CHECK_ERR
		return true;
	}

	template<typename T, typename U, typename V, typename W, typename X, typename Y>
	bool set_kernel_args(cl_kernel kernel, const T& obj1, const U& obj2, const V& obj3, const W& obj4, const X& obj5, const Y& obj6)
	{
		cl_serializer serializer(this);
		cl_int ret = clSetKernelArg(kernel, 0, sizeof(T), (void*)&obj1); CHECK_ERR
		ret = clSetKernelArg(kernel, 1, sizeof(U), (void*)&obj2); CHECK_ERR
		ret = clSetKernelArg(kernel, 2, sizeof(V), (void*)&obj3); CHECK_ERR
		ret = clSetKernelArg(kernel, 3, sizeof(W), (void*)&obj4); CHECK_ERR
		ret = clSetKernelArg(kernel, 4, sizeof(X), (void*)&obj5); CHECK_ERR
		ret = clSetKernelArg(kernel, 5, sizeof(Y), (void*)&obj6); CHECK_ERR
		return true;
	}

	template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z>
	bool set_kernel_args(cl_kernel kernel, const T& obj1, const U& obj2, const V& obj3, const W& obj4, const X& obj5, const Y& obj6, const Z& obj7)
	{
		cl_serializer serializer(this);
		cl_int ret = clSetKernelArg(kernel, 0, sizeof(T), (void*)&obj1); CHECK_ERR
		ret = clSetKernelArg(kernel, 1, sizeof(U), (void*)&obj2); CHECK_ERR
		ret = clSetKernelArg(kernel, 2, sizeof(V), (void*)&obj3); CHECK_ERR
		ret = clSetKernelArg(kernel, 3, sizeof(W), (void*)&obj4); CHECK_ERR
		ret = clSetKernelArg(kernel, 4, sizeof(X), (void*)&obj5); CHECK_ERR
		ret = clSetKernelArg(kernel, 5, sizeof(Y), (void*)&obj6); CHECK_ERR
		ret = clSetKernelArg(kernel, 6, sizeof(Z), (void*)&obj7); CHECK_ERR
		return true;
	}

	template<typename T, typename U, typename V, typename W, typename X, typename Y, typename Z, typename A>
	bool set_kernel_args(cl_kernel kernel, const T& obj1, const U& obj2, const V& obj3, const W& obj4, const X& obj5, const Y& obj6, const Z& obj7, const A& obj8)
	{
		cl_serializer serializer(this);
		cl_int ret = clSetKernelArg(kernel, 0, sizeof(T), (void*)&obj1); CHECK_ERR
		ret = clSetKernelArg(kernel, 1, sizeof(U), (void*)&obj2); CHECK_ERR
		ret = clSetKernelArg(kernel, 2, sizeof(V), (void*)&obj3); CHECK_ERR
		ret = clSetKernelArg(kernel, 3, sizeof(W), (void*)&obj4); CHECK_ERR
		ret = clSetKernelArg(kernel, 4, sizeof(X), (void*)&obj5); CHECK_ERR
		ret = clSetKernelArg(kernel, 5, sizeof(Y), (void*)&obj6); CHECK_ERR
		ret = clSetKernelArg(kernel, 6, sizeof(Z), (void*)&obj7); CHECK_ERR
		ret = clSetKernelArg(kernel, 7, sizeof(A), (void*)&obj8); CHECK_ERR
		return true;
	}
#undef CHECK_ERR

private:
	cl_device_id m_device_id = nullptr;
	cl_context m_context = nullptr;
	cl_command_queue m_command_queue = nullptr;
	cl_program m_program = nullptr;
	cl_device_fp_config m_dev_fp_config;
	
	bool m_use_mutex = false;
	std::mutex m_ocl_mutex;

	// This helper object is used to optionally serialize all calls to the CL driver after initialization.
	// Currently this is only used to work around race conditions in the Windows AMD driver.
	struct cl_serializer
	{
		inline cl_serializer(const cl_serializer&);
		cl_serializer& operator= (const cl_serializer&);

		inline cl_serializer(ocl *p) : m_p(p)
		{
			if (m_p->m_use_mutex)
				m_p->m_ocl_mutex.lock();
		}

		inline ~cl_serializer()
		{
			if (m_p->m_use_mutex)
				m_p->m_ocl_mutex.unlock();
		}

	private:
		ocl* m_p;
	};
	
	cl_image_format get_image_format(uint32_t bytes_per_pixel, bool normalized)
	{
		cl_image_format fmt;
		switch (bytes_per_pixel)
		{
		case 1: fmt.image_channel_order = CL_LUMINANCE; break;
		case 2: fmt.image_channel_order = CL_RG; break;
		case 3: fmt.image_channel_order = CL_RGB; break;
		case 4: fmt.image_channel_order = CL_RGBA; break;
		default: assert(0); fmt.image_channel_order = CL_LUMINANCE; break;
		}

		fmt.image_channel_data_type = normalized ? CL_UNORM_INT8 : CL_UNSIGNED_INT8;
		return fmt;
	}
};
