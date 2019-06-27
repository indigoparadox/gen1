
#include "gen1.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>

__global__
void setup_curand( int* init, curandState *state ) {
   int idx = threadIdx.x;

   curand_init( *init, idx, 0, &state[idx] );
}

__global__
void generate_line_c( curandState* state, uint8_t* buffer, int width ) {
   int idx = threadIdx.x;
   int i = 0;
   float f_rand;

   for( i = 0 ; width > i ; i++ ) {
      f_rand = curand_uniform( state + idx );
      f_rand *= (UINT8_MAX + 0.999999);

      buffer[i] = (uint8_t)truncf( f_rand );
   }
}

extern "C" void generate_line( uint8_t* buffer, int width ) {
   curandState *d_state = NULL;
   int init_time = 0;

   cudaMalloc( &d_state, sizeof( curandState ) * 256 );

   init_time = time( NULL );

   setup_curand<<<1, 1>>>( &init_time, d_state );

   generate_line_c<<<0, 256>>>( d_state, buffer, width );

   cudaFree( d_state );
}

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

