include(${TEST_HARNESS_DIR}/CabanaPD_TestHarness.cmake)

CabanaPD_add_tests(NAMES Particles Force Integrator)
CabanaPD_add_tests(MPI NAMES Comm)
