
#include "gen1.h"

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif /* USE_CUDA */

__global__
void fitness_score_add(
   float* out, const uint8_t* buffer_tgt,
   const uint8_t* buffer_test, int byte_width
) {
   int i = threadIdx.x;
   int stride = blockDim.x;

   for( ; byte_width > i ; i += stride ) {
      *out += abs( (buffer_tgt[i] - buffer_test[i]) );
   }
}

__global__
void combine_lines_c( uint8_t* line_dest, const uint8_t* line_src, int width ) {
   int i = threadIdx.x;
   int stride = blockDim.x;

   for( ; width > i ; i += stride ) {
      if( 0 == i % 2 ) {
         line_dest[i] = line_src[i];
      }
   }
}

extern "C" float fitness_score(
   int width, const uint8_t* buffer_tgt, const uint8_t* buffer_test
) {
   uint8_t* d_line_master = NULL;
   uint8_t* d_candidate = NULL;
   float* d_score_total = NULL;
   float out = 0;

   cudaMalloc(
      (void**)&d_score_total, sizeof( float ) );
   cudaMalloc(
      (void**)&d_candidate, sizeof( uint8_t ) * width );
   cudaMalloc(
      (void**)&d_score_total, sizeof( uint8_t ) * width );

   cudaMemcpy( d_candidate, buffer_test, width, cudaMemcpyHostToDevice );
   cudaMemcpy( d_line_master, buffer_tgt, width, cudaMemcpyHostToDevice );

   fitness_score_add<<<0, 256>>>(
      d_score_total, d_line_master, d_candidate, width );

   cudaMemcpy( &out, d_score_total, sizeof( float ), cudaMemcpyDeviceToHost );

   cudaFree( d_line_master );
   cudaFree( d_candidate );
   cudaFree( d_score_total );

   return out;
}

extern "C"
void combine_lines( uint8_t* line_dest, const uint8_t* line_src, int width ) {
   uint8_t* d_source = NULL;
   uint8_t* d_dest = NULL;

   cudaMalloc(
      (void**)&d_source, sizeof( uint8_t ) * width );
   cudaMalloc(
      (void**)&d_dest, sizeof( uint8_t ) * width );

   cudaMemcpy( d_source, line_src, width, cudaMemcpyHostToDevice );
   cudaMemcpy( d_dest, line_dest, width, cudaMemcpyHostToDevice );

   combine_lines_c<<<0, 256>>>( d_dest, d_source, width );

   cudaMemcpy(
      &line_dest, d_dest, sizeof( uint8_t ) * width, cudaMemcpyDeviceToHost );

   cudaFree( d_source );
   cudaFree( d_dest );
}

