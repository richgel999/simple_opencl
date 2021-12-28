project(simple_ocl)

cmake_minimum_required(VERSION 3.0)

if (MSVC)
	if ( CMAKE_GENERATOR_PLATFORM STREQUAL Win32 )	
		set(BUILD_X64 0)
	else()
		set(BUILD_X64 1)
	endif()
else()
	option(BUILD_X64 "build 64-bit" TRUE)
endif()

message("Initial BUILD_X64=${BUILD_X64}")
message("Initial CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

if (NOT MSVC)
	# With MSVC builds we use the Khronos lib/include files in the project's "OpenCL" directory, to completely avoid requiring fiddly to install vendor SDK's.
	# Otherwise we use the system's (if any).
	find_package( OpenCL )
	message(STATUS "OpenCL found: ${OPENCL_FOUND}")
	message(STATUS "OpenCL includes: ${OpenCL_INCLUDE_DIRS}")
	message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")
endif()

if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release )
endif()

message( ${PROJECT_NAME} " build type: " ${CMAKE_BUILD_TYPE} )

if (BUILD_X64)
	message("Building 64-bit")
else()
	message("Building 32-bit")
endif()

if (NOT MSVC)
   set(CMAKE_CXX_FLAGS -std=c++11)
   set(GCC_COMPILE_FLAGS "-fvisibility=hidden -fPIC -fno-strict-aliasing -D_LARGEFILE64_SOURCE=1 -D_FILE_OFFSET_BITS=64 -Wall -Wextra")

   if (NOT BUILD_X64)
	  set(GCC_COMPILE_FLAGS "${GCC_COMPILE_FLAGS} -m32")
   endif()

   set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_LINK_FLAGS} -Wl,-rpath .")
 
   set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${GCC_COMPILE_FLAGS} -D_DEBUG -g")
   set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GCC_COMPILE_FLAGS} -D_DEBUG -g")
endif()

set(SIMPLE_OPENCL_SRC_LIST ${COMMON_SRC_LIST} 
	src/simple_ocl.cpp
	src/ocl_device.cpp
	)

set(BIN_DIRECTORY "bin")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${BIN_DIRECTORY})

add_executable(simple_ocl ${SIMPLE_OPENCL_SRC_LIST})

if (NOT MSVC)
	# For Non-Windows builds, let cmake try and find the system OpenCL headers/libs for us.
	if (OPENCL_FOUND)
		
		target_include_directories( simple_ocl PRIVATE ${OpenCL_INCLUDE_DIRS} )
		set(SIMPLE_OPENCL_EXTRA_LIBS ${OpenCL_LIBRARIES})
	endif()

else()
	# For Windows builds, we use our local copies of the OpenCL import lib and Khronos headers.
	
	target_include_directories( simple_ocl PRIVATE "OpenCL" )

	if ( BUILD_X64 )
		target_link_libraries( simple_ocl PRIVATE "OpenCL/lib/OpenCL64" )
	else()
		target_link_libraries( simple_ocl PRIVATE "OpenCL/lib/OpenCL" )
	endif()
endif()	

if (NOT MSVC)
   target_link_libraries(simple_ocl m pthread ${SIMPLE_OPENCL_EXTRA_LIBS})
endif()

install(TARGETS simple_ocl DESTINATION bin)
	
