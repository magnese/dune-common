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

dune_add_test(SOURCES femthreadscolortest.cc
              LINK_LIBRARIES dunecommon
              SKIP_ON_77)

dune_add_test(SOURCES femthreadstest.cc
              LINK_LIBRARIES dunecommon
              SKIP_ON_77)

dune_add_test(SOURCES paralleltest.cc
              LINK_LIBRARIES dunecommon
              SKIP_ON_77)

dune_add_test(SOURCES threadstest.cc
              LINK_LIBRARIES dunecommon
              SKIP_ON_77)
