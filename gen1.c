
#include "gen1.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util.h"

void generate_line( size_t width, uint8_t* buffer ) {
   int i = 0;

   for( i = 0 ; width * PX_BYTES > i ; i++ ) {
      buffer[i] = rand() % 256;
   }
}

int fitness_score( size_t width, uint8_t* buffer_tgt, uint8_t* buffer_test ) {
   int i = 0;
   int score = 0;

   for( i = 0 ; width * PX_BYTES > i ; i++ ) {
      score += abs( (buffer_tgt[i] - buffer_test[i]) );
   }

   score /= (width * PX_BYTES);

   return UINT8_MAX - score;
}

void get_highest_2_scores_idx( int* scores, size_t scores_len, int* top2 ) {
   int i = 0;

   for( i = 0 ; scores_len > i ; i++ ) {
      if( scores[i] > scores[top2[0]] ) {
         top2[0] = i;
      } else if( scores[i] > scores[top2[1]] ) {
         top2[1] = i;
      }
   }
}

void combine_lines( size_t len, uint8_t* line_dest, const uint8_t* line_src ) {
   int i = 0;

   /* memcpy(
      provisional_candidate,
      &(line->candidates[top2_idx[1]]),
      (line->width * PX_BYTES) / 2
   ); */

   for( i = 0 ; len > i ; i++ ) {
      if( 0 == i % 2 ) {
         line_dest[i] = line_src[i];
      }
   }
}

void* evolve_thread( void* line_raw ) {
   int scores[MAX_CANDIDATES] = { 0 };
   int top2_idx[2] = { 0 };
   uint8_t* candidate = NULL;
   uint8_t* top_candidate = NULL;
   uint8_t* provisional_candidate = NULL;
   struct line* line = (struct line*)line_raw;
   int i = 0, j = 0, generation = 0, provisional_score = 0,
      high_provisional_score = 0, high_provisional_idx = 0;

   provisional_candidate = calloc( line->width * PX_BYTES, sizeof( uint8_t ) );

   for( generation = 0 ; line->generations > generation ; generation++ ) {
      high_provisional_score = 0;
      high_provisional_idx = 0;

      for( i = 0 ; MAX_CANDIDATES > i ; i++ ) {
         /* Get the score for this candidate. */
         candidate = &(line->candidates[i]);
         scores[i] = fitness_score( line->width, line->line_master, candidate );
      }

      /* Combine the two highest-scored items. */
      get_highest_2_scores_idx( scores, MAX_CANDIDATES, top2_idx );
      log_out( "[T%d-G%d/%d] Top 2 scores: %d, %d\n", 
         line->idx, generation, line->generations, top2_idx[0], top2_idx[1] );

      /* Create provisional candidates and get their scores. */
      for( i = 0 ; MAX_CANDIDATES > i ; i++ ) {
         memcpy(
            provisional_candidate, 
            &(line->candidates[top2_idx[0]]),
            line->width * PX_BYTES
         );
         combine_lines(
            line->width, provisional_candidate,
            &(line->candidates[i]) );
         provisional_score = fitness_score( line->width, line->line_master,
            provisional_candidate );
         if( provisional_score > high_provisional_score ) {
            high_provisional_idx = i;
            high_provisional_score = provisional_score;
         }
      }

      /* Go with the highest provisional score. */
      log_out(
         "[T%d-G%d/%d] Highest provisional score (idx): %d (%d)\n", 
         line->idx, generation, line->generations, 
         high_provisional_score, high_provisional_idx );
      memcpy(
         provisional_candidate, 
         &(line->candidates[top2_idx[0]]),
         line->width * PX_BYTES
      );
      combine_lines(
         line->width, provisional_candidate,
         &(line->candidates[high_provisional_idx]) );
#if 0
         if( provisional_score < scores[top2_idx[0]] ) {
            log_out( "[T%d-G%d/%d] Provisional score %d below %d, skipping...",
               line->idx, generation, line->generations,
               provisional_score, scores[top2_idx[0]] );
            generation--;
            continue;
         }
#endif

      /* Copy provisional candidate to first candidate slot. */
      memcpy(
         &(line->candidates[0]),
         provisional_candidate,
         line->width * PX_BYTES
      );

      /* Generate new lines based on new top score. */
      top_candidate = &(line->candidates[0]);
      for( i = 1 ; MAX_CANDIDATES > i ; i++ ) {
         candidate = &(line->candidates[i]);
         for( j = 0 ; line->width * PX_BYTES > j ; j++ ) {
            /* Subtle mutation per pixel. */
            if( 0 == rand() / 10 ) {
               candidate[j] = top_candidate[j] + (rand() % 3);
            } else {
               candidate[j] = top_candidate[j] - (rand() % 3);
            }
         }
      }
   }

   top_candidate = &(line->candidates[top2_idx[0]]);
   memcpy( &(line->line_blank[0]), top_candidate, line->width * PX_BYTES );

   free( provisional_candidate );
   free( line->candidates );

   return NULL;
}

