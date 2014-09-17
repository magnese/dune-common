add_subdirectory("test" EXCLUDE_FROM_ALL)

#install headers
install(FILES
        collectivecommunication.hh
        communicator.hh
        indexset.hh
        indicessyncer.hh
        interface.hh
        localindex.hh
        mpicollectivecommunication.hh
        mpicommunicatorparadigm.hh
        mpiguard.hh
        mpihelper.hh
        mpiparallelparadigm.hh
        mpitraits.hh
        plocalindex.hh
        remoteindices.hh
        selection.hh
        threadcommunicatorparadigm.hh
        threadparallelparadigm.hh
        variablesizecommunicator.hh
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dune/common/parallel)

