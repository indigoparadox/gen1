
#include <check.h>

#include "../src/gen1.h"
#include "../src/util.h"

const uint8_t g_test_str_1[10] = { 2, 3, 4, 5, 6, 7, 8, 9, 2, 4 };
const uint8_t g_test_str_2[10] = { 8, 7, 6, 5, 4, 3, 6, 5, 4, 3 };
const uint8_t g_test_scomb[10] = { 2, 7, 4, 5, 6, 3, 8, 5, 2, 3 };

START_TEST( check_combine_lines ) {
   uint8_t local_test[10];

   memcpy( local_test, g_test_str_1, 10 );

   combine_lines( 10, local_test, g_test_str_2 );
   
   if( 0 == _i % 2 ) {
      ck_assert_int_eq( local_test[_i], g_test_str_2[_i] );
   } else {
      ck_assert_int_eq( local_test[_i], g_test_str_1[_i] );
   }
}
END_TEST

Suite* gen_suite( void ) {
   Suite* s;
   TCase* tc_core;

   s = suite_create( "gen" );

   /* Test Cases */
   tc_core = tcase_create( "core" );
   
   tcase_add_loop_test( tc_core, check_combine_lines, 0, 10 );

   suite_add_tcase( s, tc_core );

   return s;
}

