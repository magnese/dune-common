set(MPITESTPROGS indicestest indexsettest paralleltest syncertest selectiontest threadstest variablesizecommunicatortest)

add_directory_test_target(_test_target)
# We do not want want to build the tests during make all,
# but just build them on demand
add_dependencies(${_test_target} ${MPITESTPROGS})

add_executable("indexsettest" indexsettest.cc)
target_link_libraries("indexsettest" "dunecommon" ${CMAKE_THREAD_LIBS_INIT} ${})

add_executable("threadstest" threadstest.cc)
target_link_libraries("threadstest" "dunecommon")

include(DuneMPI)
add_executable("indicestest" indicestest.cc)
target_link_libraries("indicestest" "dunecommon")
add_dune_mpi_flags(indicestest)

add_executable("paralleltest" paralleltest.cc)
target_link_libraries("paralleltest" "dunecommon")
add_dune_mpi_flags(paralleltest)

add_executable("selectiontest" selectiontest.cc)
target_link_libraries("selectiontest" "dunecommon")
add_dune_mpi_flags(selectiontest)

add_executable("syncertest" syncertest.cc)
target_link_libraries("syncertest" "dunecommon")
add_dune_mpi_flags(syncertest)

add_executable("variablesizecommunicatortest" variablesizecommunicatortest.cc)
target_link_libraries("variablesizecommunicatortest" "dunecommon")
add_dune_mpi_flags(variablesizecommunicatortest)

add_test(indexsettest indexsettest)
add_test(selectiontest selectiontest)
add_test(indicestest indicestest)
add_test(paralleltest paralleltest)
add_test(syncertest syncertest)
add_test(threadstest threadstest)
add_test(variablesizecommunicatortest variablesizecommunicatortest)

# treat tests returning code 77 as skipped
set_tests_properties(
  indicestest
  paralleltest
  syncertest
  threadstest
  variablesizecommunicatortest
  PROPERTIES SKIP_RETURN_CODE 77)
