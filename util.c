
#include "util.h"

#include <stdio.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <assert.h>

pthread_mutex_t log_mutex = PTHREAD_MUTEX_INITIALIZER;
FILE* log_file = NULL;

void log_open( const char* logname ) {
   if( NULL != logname ) {
      log_file = fopen( logname, "w" );
      assert( NULL != log_file );
   }
}

void log_close() {
   if( NULL != log_file ) {
      fclose( log_file );
   }
}

void log_out( const char* msg, ... ) {
   va_list args;

   if( NULL == log_file ) {
      return;
   }
  
   va_start( args, msg );

   /* Lock the log file before writing to it. */
   pthread_mutex_lock( &log_mutex );
   vfprintf( log_file, msg, args );
   pthread_mutex_unlock( &log_mutex );

   va_end( args );
}

int open_bmp(
   char* filename, uint8_t** map, struct bmp_info** info, uint8_t** bmp_data
) {
   struct bmp_header* header = NULL;
   struct stat st;
   size_t filesize = 0;
   int bmp_handle = 0;

   stat( filename, &st );
   filesize = st.st_size;

   bmp_handle = open( filename, O_RDONLY, 0 );
   assert( 0 < bmp_handle );

   *map = mmap( NULL, filesize, PROT_READ, MAP_PRIVATE | MAP_POPULATE,
      bmp_handle, 0 );
   assert( MAP_FAILED != *map );

   header = (struct bmp_header*)(*map);
   *info = (struct bmp_info*)&((*map)[sizeof( struct bmp_header )]);

   *bmp_data = (uint8_t*)&((*map)[sizeof( struct bmp_header ) +
      sizeof( struct bmp_info )]);

   assert( 0x4d42 == header->type );
   assert( 40 == (*info)->size );
   assert( 1 == (*info)->planes );
   assert( 0 == (*info)->compression );
   assert( 0 == (*info)->colors );
   assert( 24 == (*info)->bpp );
   assert( 0 == ((*info)->bpp * (*info)->width) % 4 );
   assert( 0 < (*info)->height );

   if( 0 < bmp_handle ) {
      close( bmp_handle );
   }

   if( NULL == *map ) {
      return 0;
   } else {
      return filesize;
   }
}

