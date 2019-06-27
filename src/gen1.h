
#ifndef GEN1_H
#define GEN1_H

#include <stdint.h>
#include <stddef.h>

#define MAX_CANDIDATES 6
#define MAX_GENERATIONS 10000

struct line {
   size_t byte_width;
   int idx;
   uint8_t* line_master;
   uint8_t* line_blank;
   uint8_t* candidates;
   size_t generations;
};

#ifndef USE_CUDA
void generate_line( size_t width, uint8_t* buffer );
float fitness_score(
   int width, const uint8_t* buffer_tgt, const uint8_t* buffer_test );
void combine_lines( uint8_t* line_dest, const uint8_t* line_src, int width );
#endif /* USE_CUDA */
void get_highest_2_scores_idx( int* scores, size_t scores_len, int* top2 );
void* evolve_thread( void* line_raw );

#endif /* GEN1_H */

