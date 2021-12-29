# simple_opencl
This is a simple and *practical* C++ sample showing how to use OpenCL v1.2 on Windows/Linux/OSX with no 3rd party SDK installs required under Windows/OSX.

Unlike every other OpenCL example I've seen, this example demonstrates a bunch of things you would need to do in practice to build an app using OpenCL:

- Importantly, *no 3rd party SDK dependencies are required to compile/link under Windows*. All required OpenCL headers and the 2 import .LIB's are in the "OpenCL" directory.
- How to safely use OpenCL from multiple threads (by using a local context, and creating your command queue/kernels on that context).
- How to work around AMD driver serialization issues if you use OpenCL from multiple threads
- How to load your program kernel source code from either a file, or from an array in a C-style header

Windows has strong support for OpenCL v1.2 on NVidia, AMD, and Intel drivers. In my testing, even brand new Windows AMD machines right out of the box with no updates have working OpenCL drivers. Some of OpenCL v1.2's strengths are its maturity, driver support, ease of use, and no large 3rd party SDK's or libraries are required to use it. Here's a good [introductory book](https://www.amazon.com/gp/product/B097827WWG/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1) on OpenCL.

### Building

Use "cmake .". Then under OSX/Linux, use "make".

Under Windows load the generated .SLN with Visual Studio 2019/2022. All included headers/import libs are in the project, so no 3rd party SDK's are required. Be sure to right click on "simple_ocl" and select "Set as Startup Project" before running.

Under Linux, you will need a driver with OpenCL support, and the OpenCL headers/libraries. The easiest thing to do is to use the NVidia proprietary driver, then use "sudo apt-get install nvidia-cuda-toolkit". [This](https://linuxhandbook.com/setup-opencl-linux-docker/) page may help. Install and run the "clinfo" app to validate that your driver supports OpenCL. CMake will automatically find the OpenCL headers/libraries.

Under OSX (High Sierra), it just works for me. CMake handles finding the libs/headers.

### Running

The "bin" directory contains the sample executable, "simple_ocl". Running it will generate a buffer of random values. The GPU kernel will modify this buffer, and the CPU will read it back and validate its contents.

You should see something like this (note the random numbers will likely be different for you):

```
OpenCL platform version: "OpenCL 3.0 CUDA 11.4.94"
Serializing OpenCL calls across threads: 0
OpenCL device initialized successfully
Using kernel source code from array in header src/ocl_kernels.h
OpenCL context initialized successfully
Running "process_buffer" kernel
Validation succeeded
Input/output buffer contents (first 16 bytes):
41 41
35 34
190 188
132 135
225 229
108 105
214 208
174 169
82 90
144 153
73 67
241 250
241 253
187 182
233 231
235 228
```

### Modifying the kernel source code

By default, this sample compiles the OpenCL program from an array of text in [src/ocl_kernels.h](src/ocl_kernels.h). This header file was created using the "xxd" tool with the -i option from the kernel source code file located under [bin/ocl_kernels.cl](bin/ocl_kernels.cl). If you want the sample to always load the kernel source code from the "bin" directory instead, set `OCL_USE_KERNELS_HEADER` to 0 in [src/ocl_device.cpp](https://github.com/richgel999/simple_opencl/blob/main/src/ocl_device.cpp).

### Design

This sample was derived from how we're using OpenCL in Basis Universal, our GPU texture interchange library/tool.

[simple_ocl_wrapper.h](src/simple_ocl_wrapper.h) contains a basic C++ wrapper on top of the C OpenCL API. OpenCL includes its own wrapper, but by writing your own you can control exactly how OpenCL is called, which features are exposed, and what C++ features are utilized by the wrapper.

[ocl_device.cpp/h](src/ocl_device.h) uses this wrapper to create the OpenCL device. It exposes a simple C-style API that callers can use to initialize/deinitalize the device, and create/destroy per-thread contexts and kernels. Out of the box it supports a single kernel source code file (which can contain multiple kernels) which can be either loaded from disk or from a C-style array in a header file.

[simple_ocp.cpp] utilizes the C-style API exposed by ocl_device.h. It creates a byte buffer of random numbers, then calls `opencl_process_buffer()` in ocl_device.cpp.
