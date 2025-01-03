add_executable(ElasticWave elastic_wave.cpp)
target_link_libraries(ElasticWave LINK_PUBLIC CabanaPD)
install(TARGETS ElasticWave DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(KalthoffWinkler kalthoff_winkler.cpp)
target_link_libraries(KalthoffWinkler LINK_PUBLIC CabanaPD)
install(TARGETS KalthoffWinkler DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(CrackBranching crack_branching.cpp)
target_link_libraries(CrackBranching LINK_PUBLIC CabanaPD)
install(TARGETS CrackBranching DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(FragmentingCylinder fragmenting_cylinder.cpp)
target_link_libraries(FragmentingCylinder LINK_PUBLIC CabanaPD)
install(TARGETS FragmentingCylinder DESTINATION ${CMAKE_INSTALL_BINDIR})

if(Halyard_ENABLE_TESTING)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/inputs/fragmenting_cylinder.json
        ${CMAKE_CURRENT_BINARY_DIR}/fragmenting_cylinder.json
        COPYONLY
    )
  
  add_test(NAME CabanaPD_RegressionTest_FragmentingCylinder COMMAND ${NONMPI_PRECOMMAND} $<TARGET_FILE:FragmentingCylinder> fragmenting_cylinder.json)
endif()
