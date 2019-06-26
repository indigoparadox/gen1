
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

   return 255 - score;
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

void combine_lines( uint8_t* line_dest, const uint8_t* line_src ) {

}

void* evolve_thread( void* line_raw ) {
   int scores[MAX_CANDIDATES] = { 0 };
   int top2_idx[2] = { 0 };
   uint8_t* candidate = NULL;
   uint8_t* top_candidate = NULL;
   uint8_t* provisional_candidate = NULL;
   struct line* line = (struct line*)line_raw;
   int i = 0, j = 0, generation = 0, provisional_score = 0;

   provisional_candidate = calloc( line->width * PX_BYTES, sizeof( uint8_t ) );

   for( generation = 0 ; line->generations > generation ; generation++ ) {
      for( i = 0 ; MAX_CANDIDATES > i ; i++ ) {
         /* Get the score for this candidate. */
         candidate = &(line->candidates[i]);
         scores[i] = fitness_score( line->width, line->line_master, candidate );
      }

      /* Combine the two highest-scored items. */
      get_highest_2_scores_idx( scores, MAX_CANDIDATES, top2_idx );
      log_out( "[T%d-G%d/%d] Top 2 scores: %d, %d\n", 
         line->idx, generation, line->generations, top2_idx[0], top2_idx[1] );

      /* Create a provisional candidate and get its score. */
      memcpy(
         provisional_candidate,
         &(line->candidates[top2_idx[0]]),
         line->width * PX_BYTES
      );
      memcpy(
         provisional_candidate,
         &(line->candidates[top2_idx[1]]),
         (line->width * PX_BYTES) / 2
      );
      provisional_score = fitness_score( line->width, line->line_master,
         provisional_candidate );

      if( provisional_score < scores[top2_idx[0]] ) {
         log_out( "[T%d-G%d/%d] Provisional score %d below %d, skipping...",
            line->idx, generation, line->generations,
            provisional_score, scores[top2_idx[0]] );
         generation--;
         continue;
      }

      /* Copy provisional candidate to top score. */
      memcpy(
         &(line->candidates[top2_idx[0]]),
         provisional_candidate,
         line->width * PX_BYTES
      );

      /* Generate new lines based on new top score. */
      top_candidate = &(line->candidates[top2_idx[0]]);
      for( i = 1 ; MAX_CANDIDATES > i ; i++ ) {
         candidate = &(line->candidates[i]);
         for( j = 0 ; line->width * PX_BYTES > j ; j++ ) {
            /* Subtle mutation per pixel. */
            candidate[j] = top_candidate[j] + (rand() % 3);
         }
      }
   }

   top_candidate = &(line->candidates[top2_idx[0]]);
   memcpy( &(line->line_blank[0]), top_candidate, line->width * PX_BYTES );

   free( provisional_candidate );
   free( line->candidates );

   return NULL;
}

