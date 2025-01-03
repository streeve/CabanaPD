configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/inputs/hertzian_contact.json
  ${CMAKE_CURRENT_BINARY_DIR}/hertzian_contact.json
  COPYONLY
)

include(${TEST_HARNESS_DIR}/CabanaPD_TestHarness.cmake)

CabanaPD_add_tests(REGRESSION NAMES Hertz)
