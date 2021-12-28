//#define _DEBUG

#ifndef NULL
	#define NULL 0L
#endif

typedef char int8_t;
typedef uchar uint8_t;

typedef short int16_t;
typedef ushort uint16_t;

typedef int int32_t;
typedef uint uint32_t;

typedef long int64_t;
typedef ulong uint64_t;

typedef uchar4 color_rgba;

#define UINT32_MAX 0xFFFFFFFFUL
#define INT64_MAX LONG_MAX
#define UINT64_MAX ULONG_MAX

#ifdef _DEBUG
	// printf() works on NVidia's driver, not sure about others.
	inline void internal_assert(bool x, constant char *pMsg, int line)
	{
		if (!x)
			printf("assert() failed on line %i: %s\n", line, pMsg);
	}
	#define assert(x) internal_assert(x, #x, __LINE__)
#else
	#define assert(x)
#endif

kernel void process_buffer(
    const global uint8_t *pInput_buf,
	global uint8_t *pOutput_buf,
    uint32_t buf_size)
{
	const uint32_t buf_ofs = get_global_id(0);

	assert(buf_ofs < buf_size);
	
	pOutput_buf[buf_ofs] = pInput_buf[buf_ofs] ^ (uint8_t)buf_ofs;
}
