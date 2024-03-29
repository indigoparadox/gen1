
#include "gen1.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif /* USE_CUDA */

#include "util.h"

#ifndef USE_CUDA
void generate_line( int width, uint8_t* buffer ) {
   int i = 0;

   for( i = 0 ; width > i ; i++ ) {
      buffer[i] = rand() % 256;
   }
}

score_t fitness_score(
   int width, const uint8_t* buffer_tgt, const uint8_t* buffer_test
) {
   int i = 0;
   score_t score = 0;

   for( i = 0 ; width * PX_BYTES > i ; i++ ) {
      /* score += abs( (buffer_tgt[i] - buffer_test[i]) ); */
      if( buffer_test[i] == buffer_tgt[i] ) {
         score++;
      }
   }

   //score /= (width * PX_BYTES);

   return score;
}

void combine_lines( uint8_t* line_dest, const uint8_t* line_src, int width ) {
   int i = 0;

   /* memcpy(
      provisional_candidate,
      &(line->candidates[top2_idx[1]]),
      (line->width * PX_BYTES) / 2
   ); */

   for( i = 0 ; width > i ; i++ ) {
      if( 0 == i % 2 ) {
         line_dest[i] = line_src[i];
      }
   }
}

#endif /* !USE_CUDA */

void* evolve_thread( void* line_raw ) {
   score_t* scores = NULL;
   uint8_t* candidate = NULL;
   uint8_t* top_candidate = NULL;
   uint8_t* provisional_candidate = NULL;
   struct line* line = (struct line*)line_raw;
   int i = 0, j = 0, generation = 0,
      high_provisional_idx = 0,
      top_score_idx = 0;
   score_t provisional_score = 0, high_provisional_score = 0;

   provisional_candidate = calloc( line->byte_width, sizeof( uint8_t ) );
   scores = calloc( line->candidates_sz, sizeof( score_t ) );

   for( generation = 0 ; line->generations > generation ; generation++ ) {
      high_provisional_score = 0;
      high_provisional_idx = 0;

      for( i = 0 ; line->candidates_sz > i ; i++ ) {
         /* Get the score for this candidate. */
         candidate = &(line->candidates[i]);

         scores[i] = fitness_score( line->byte_width, line->line_master,
            candidate );
         if( scores[i] > scores[top_score_idx] ) {
            top_score_idx = i;
         }
      }

      log_out( "[T%dx%d-G%d/%d] Top score: %d (%d)\n", 
         line->x, line->y, generation, line->generations,
         scores[top_score_idx], top_score_idx );
      csv_out( "%d,%d,%d,%d\n", line->x, line->y,
         generation, scores[top_score_idx] );

      /* Create provisional candidates and get their scores. */
      for( i = 0 ; line->candidates_sz > i ; i++ ) {
         memcpy(
            provisional_candidate, 
            &(line->candidates[top_score_idx]),
            line->byte_width
         );
         combine_lines(
            provisional_candidate,
            &(line->candidates[i]), line->byte_width );
         provisional_score = fitness_score( line->byte_width, line->line_master,
            provisional_candidate );
         if( provisional_score > high_provisional_score ) {
            high_provisional_idx = i;
            high_provisional_score = provisional_score;
         }
      }

      /* Go with the highest provisional score. */
      log_out(
         "[T%dx%d-G%d/%d] Highest provisional score (idx): %d (%d)\n", 
         line->x, line->y, generation, line->generations, 
         high_provisional_score, high_provisional_idx );
      memcpy(
         provisional_candidate, 
         &(line->candidates[top_score_idx]),
         line->byte_width
      );
      combine_lines(
         provisional_candidate,
         &(line->candidates[high_provisional_idx]),
         line->byte_width );

      /* Copy provisional candidate to first candidate slot. */
      memcpy(
         &(line->candidates[0]),
         provisional_candidate,
         line->byte_width
      );

      /* Generate new lines based on new top score. */
      top_candidate = &(line->candidates[0]);
      for( i = 1 ; line->candidates_sz > i ; i++ ) {
         candidate = &(line->candidates[i]);
         for( j = 0 ; line->byte_width > j ; j++ ) {
            /* Subtle mutation per pixel. */
            if( 0 == rand() % 150 ) {
               candidate[j] = top_candidate[j] + (rand() % 10);
            } else {
               candidate[j] = top_candidate[j] - (rand() % 10);
            }
         }
      }

      /* Generate some radical new lines. */
      for( i = line->candidates_sz / 2 ; line->candidates_sz > i ; i++ ) {
         //generate_line( line->byte_width, &(line->candidates[i]) );
      }
   }

   top_candidate = &(line->candidates[top_score_idx]);
   memcpy( &(line->line_blank[0]), top_candidate, line->byte_width );

   free( provisional_candidate );
   free( line->candidates );
   free( scores );

   return NULL;
}

