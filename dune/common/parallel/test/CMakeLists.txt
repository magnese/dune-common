dune_add_test(SOURCES indexsettest.cc
              LINK_LIBRARIES dunecommon)

dune_add_test(SOURCES indicestest.cc
              LINK_LIBRARIES dunecommon
              SKIP_ON_77)

dune_add_test(SOURCES selectiontest.cc
              LINK_LIBRARIES dunecommon)

dune_add_test(SOURCES syncertest.cc
              LINK_LIBRARIES dunecommon
              SKIP_ON_77)

dune_add_test(SOURCES variablesizecommunicatortest.cc
              SKIP_ON_77)
