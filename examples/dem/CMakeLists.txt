add_executable(PowderFill powder_fill.cpp)
target_link_libraries(PowderFill LINK_PUBLIC CabanaPD)
install(TARGETS PowderFill DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(PowderHeap powder_heap.cpp)
target_link_libraries(PowderHeap LINK_PUBLIC CabanaPD)
install(TARGETS PowderHeap DESTINATION ${CMAKE_INSTALL_BINDIR})

add_executable(PowderRecoater powder_recoater.cpp)
target_link_libraries(PowderRecoater LINK_PUBLIC CabanaPD)
install(TARGETS PowderRecoater DESTINATION ${CMAKE_INSTALL_BINDIR})
