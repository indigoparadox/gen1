
#ifndef GEN1_H
#define GEN1_H

#include <stdint.h>
#include <stddef.h>

#define MAX_CANDIDATES 6
#define MAX_GENERATIONS 10000

struct line {
   size_t width;
   int idx;
   uint8_t* line_master;
   uint8_t* line_blank;
   uint8_t* candidates;
   size_t generations;
};

void generate_line( size_t width, uint8_t* buffer );
int fitness_score( size_t width, uint8_t* buffer_tgt, uint8_t* buffer_test );
void get_highest_2_scores_idx( int* scores, size_t scores_len, int* top2 );
void* evolve_thread( void* line_raw );

#endif /* GEN1_H */

