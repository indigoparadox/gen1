
#ifndef UTIL_H
#define UTIL_H

#include <stdarg.h>
#include <stdint.h>

#define PX_BYTES 3

struct bmp_header {
   uint16_t type;
   uint32_t size;
   uint16_t reserved1;
   uint16_t reserved2;
   uint32_t offset;
} __attribute__( (packed) );

struct bmp_info {
   uint32_t size;
   int32_t width;
   int32_t height;
   uint16_t planes;
   uint16_t bpp;
   uint32_t compression;
   uint32_t img_size;
   int32_t xppm;
   int32_t yppm;
   uint32_t colors;
   uint32_t colors_important;
} __attribute__( (packed) );

void log_open( const char* logname );
void log_close();
void log_out( const char* msg, ... );
int open_bmp(
   char* filename, uint8_t** map, struct bmp_info** info, uint8_t** bmp_data );

#endif /* UTIL_H */

