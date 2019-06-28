
#include "gen1.h"
#include "util.h"

#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>

#define MAX_CANDIDATES 20
#define MAX_GENERATIONS 10000
#define CHUNK_COUNT 16

int main( int argc, char** argv ) {
   struct bmp_info* info = NULL;
   uint8_t* bmp_data = NULL;
   uint8_t* bmp_map = NULL;
   int x, y, i, j, k, img_byte_width, img_chunk_width;
   size_t filesize;
   uint8_t* canvas_whole = NULL;
   uint8_t* canvas = NULL;
   pthread_t** threads = NULL;
   struct line* line = NULL;
   FILE* canvas_file = NULL;
   int c = 0;
   char* filename = "home.bmp";
   char* logname = NULL;
   char* outname = "out.bmp";
   size_t generations = MAX_GENERATIONS;
   size_t candidates = MAX_CANDIDATES;
   size_t chunks = CHUNK_COUNT;
   int start_line = 0;
   int end_line = 0;
   char* csvname = NULL;

   while( (c = getopt( argc, argv, "b:l:g:s:e:o:t:c:a:k:" )) != -1 ) {
      switch( c ) {
      case 'b':
         filename = optarg;
         break;

      case 'c':
         csvname = optarg;
         break;

      case 'l':
         logname = optarg;
         break;

      case 'o':
         outname = optarg;
         break;

      case 'g':
         generations = atoi( optarg );
         break;

      case 'a':
         candidates = atoi( optarg );
         break;

      case 'k':
         chunks = atoi( optarg );
         break;

      case 's':
         start_line = atoi( optarg );
         break;

      case 'e':
         end_line = atoi( optarg );
         break;
      }
   }

   assert( NULL != filename );
  
   srand( time( NULL ) );

   log_open( logname );
   csv_open( csvname );

   filesize = open_bmp( filename, &bmp_map, &info, &bmp_data );

   printf(
      "Size:\t%d\nWidth:\t%d\nHeight\t%d\nColors:\t%d\n"
      "XPPM:\t%d\nYPPM:\t%d\nBPP:\t%d\n\nGenerations:\t%ld\n\n",
      info->size, info->width, info->height, info->colors,
      info->xppm, info->yppm, info->bpp, generations );

   log_out(
      "Size:\t%d\nWidth:\t%d\nHeight\t%d\nColors:\t%d\n"
      "XPPM:\t%d\nYPPM:\t%d\nBPP:\t%d\n\n",
      info->size, info->width, info->height, info->colors,
      info->xppm, info->yppm,info->bpp );

   /* Allocate an output bitmap of the same size as the input. */
   canvas_whole = calloc( filesize, sizeof( uint8_t ) );
   assert( NULL != canvas_whole );
   canvas = &(canvas_whole[
      sizeof( struct bmp_header ) + sizeof( struct bmp_info )]);

   /* Copy the headers to the new bitmap. */
   memcpy( canvas_whole, bmp_map, sizeof( struct bmp_header ) );
   memcpy( &(canvas_whole[sizeof( struct bmp_header )]),
      &(bmp_map[sizeof( struct bmp_header )]), sizeof( struct bmp_info ) );

   /* Allocate a thread pool. */
   threads = calloc( info->height, sizeof( pthread_t* ) );
   for( i = 0 ; info->height > i ; i++ ) {
      threads[i] = calloc( chunks, sizeof( pthread_t ) );
   }

   assert( start_line <= end_line );
   assert( start_line >= 0 );
   assert( end_line >= 0 );

   if( 0 == end_line ) {
      end_line = info->height - 1;
   }

   csv_out( "Thread,Generation,TopScore\n" );
   img_byte_width = info->width * PX_BYTES;
   img_chunk_width = img_byte_width / chunks;
   assert( 0 == img_byte_width % img_chunk_width );
   for( y = end_line ; start_line <= y ; y-- ) {
      for( x = 0 ; chunks > x ; x++ ) {
         i = (y * img_byte_width) + (x * img_chunk_width);

         line = calloc( 1, sizeof( struct line ) );
         line->byte_width = img_chunk_width;
         line->y = y;
         line->x = x * img_chunk_width;
         line->candidates_sz = candidates;
         line->generations = generations;
         line->line_master = &(bmp_data[i]);
         line->line_blank = &(canvas[i]);
         line->candidates = calloc( candidates, img_chunk_width );
         for( j = 0 ; candidates > j ; j++ ) {
            generate_line( line->byte_width,
               &(line->candidates[j * img_chunk_width]) );
         }

         pthread_create( &(threads[y][x]), NULL, evolve_thread, (void*)line );
      }
   }

   for( y = end_line ; start_line <= y ; y-- ) {
      for( x = 0 ; chunks > x ; x++ ) {
         pthread_join( threads[y][x], NULL );
      }
   }

   /* Write the final bitmap to disk. */
   canvas_file = fopen( outname, "wb" );
   assert( NULL != canvas_file );
   fwrite( canvas_whole, sizeof( uint8_t ), filesize, canvas_file );
   fclose( canvas_file );

   if( NULL != bmp_data ) {
      munmap( bmp_map, filesize );
   }

   if( NULL != canvas_whole ) {
      free( canvas_whole );
   }

   if( NULL != threads ) {
      free( threads );
   }

   log_close();

   return 0;
}

