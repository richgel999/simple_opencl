# simple_opencl
This is a simple and *practical* C++ sample showing how to use OpenCL v1.2 on Windows/Linux/OSX with no 3rd party SDK installs required under Windows/OSX.

Unlike every other OpenCL example I've seen, this example demonstrates a bunch of things you would need to do in practice to ship an app using OpenCL:

- Importantly, *no 3rd party SDK dependencies are required to compile/link under Windows*. All required OpenCL headers and the 2 import .LIB's are in the "OpenCL" directory.
- How to safely use OpenCL from multiple threads (by using a local context, and creating your command queue/kernels on that context).
- How to work around AMD driver serialization issues if you use OpenCL from multiple threads
- How to load your program source from either a file, or from an array in a C-style header

Windows has strong support for OpenCL v1.2 on NVidia, AMD, and Intel drivers. In my testing, even brand new Windows AMD machines right out of the box with no updates have working OpenCL drivers.

### Building

Use "cmake .". 

Under Windows load the generated .SLN with Visual Studio 2019/2022. All included headers/import libs are in the project, so no 3rd party SDK's are required.

Under Linux, you will need a driver with OpenCL support, and the OpenCL headers/libraries. The easiest thing to do is to use the NVidia proprietary driver, then use "sudo apt-get install nvidia-cuda-toolkit". [This](https://linuxhandbook.com/setup-opencl-linux-docker/) page may help. CMake will automatically find the OpenCL headers/libraries.

Under OSX (High Sierra), it just works for me. CMake handles finding the libs/headers.

### Running

The "bin" directory will contain the executable. Running it will generate a buffer of random values. The GPU kernel will modify this buffer, and the CPU will read it back and validate its contents.
