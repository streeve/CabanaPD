add_executable(ThermalDeformation thermal_deformation.cpp)
target_link_libraries(ThermalDeformation LINK_PUBLIC CabanaPD)

add_executable(ThermalCrack thermal_crack.cpp)
target_link_libraries(ThermalCrack LINK_PUBLIC CabanaPD)

add_executable(Monoblock thermal_monoblock_deformation.cpp)
target_link_libraries(Monoblock LINK_PUBLIC CabanaPD)

install(TARGETS ThermalDeformation ThermalCrack Monoblock DESTINATION ${CMAKE_INSTALL_BINDIR})
