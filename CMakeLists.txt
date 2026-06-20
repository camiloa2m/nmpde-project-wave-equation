cmake_minimum_required(VERSION 3.12.0)
project(wave_equation LANGUAGES CXX C)

include(../common/cmake-common.cmake)

add_executable(exercise-wave src/exercise-wave.cpp src/WaveEquation.cpp)
deal_ii_setup_target(exercise-wave)

add_executable(convergence-wave src/convergence-wave.cpp src/WaveEquation.cpp)
deal_ii_setup_target(convergence-wave)

add_executable(energy-wave src/energy-wave.cpp src/WaveEquation.cpp)
deal_ii_setup_target(energy-wave)
