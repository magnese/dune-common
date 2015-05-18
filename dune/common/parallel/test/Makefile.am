# $Id$

MPITESTS = indicestest                        \
           indexsettest                       \
           syncertest                         \
           selectiontest                      \
           variablesizecommunicatortest       \
           mpiremoteindicestest               \
           threadremoteindicestest

# which tests where program to build and run are equal
NORMALTESTS =

# list of tests to run (indicestest is special case)
TESTS = $(NORMALTESTS) $(MPITESTS)

# programs just to build when "make check" is used
check_PROGRAMS = $(NORMALTESTS) $(MPITESTS)

# define the programs
indicestest_SOURCES = indicestest.cc
indicestest_CPPFLAGS = $(AM_CPPFLAGS)	\
	$(DUNEMPICPPFLAGS)
indicestest_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)
indicestest_LDADD =										\
	$(DUNEMPILIBS)											\
	$(LDADD)

selectiontest_SOURCES = selectiontest.cc
selectiontest_CPPFLAGS = $(AM_CPPFLAGS)	\
	$(DUNEMPICPPFLAGS)
selectiontest_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)
selectiontest_LDADD =										\
	$(DUNEMPILIBS)												\
	$(LDADD)

indexsettest_SOURCES = indexsettest.cc

syncertest_SOURCES = syncertest.cc
syncertest_CPPFLAGS = $(AM_CPPFLAGS)					\
	$(DUNEMPICPPFLAGS)													\
	$(DUNE_COMMON_CPPFLAGS)
syncertest_LDFLAGS = $(AM_LDFLAGS)						\
	$(DUNEMPILDFLAGS)														\
	$(DUNE_COMMON_LDFLAGS)
syncertest_LDADD =														\
	$(DUNE_COMMON_LDFLAGS) $(DUNE_COMMON_LIBS)	\
	$(DUNEMPILIBS)															\
	$(LDADD)


variablesizecommunicatortest_SOURCES = variablesizecommunicatortest.cc
variablesizecommunicatortest_CPPFLAGS = $(AM_CPPFLAGS)	\
	$(DUNEMPICPPFLAGS)			                       				\
	$(DUNE_COMMON_CPPFLAGS)
variablesizecommunicatortest_LDFLAGS = $(AM_LDFLAGS)		\
	$(DUNEMPILDFLAGS)			                       					\
	$(DUNE_COMMON_LDFLAGS)
variablesizecommunicatortest_LDADD =                    \
	$(DUNE_COMMON_LDFLAGS) $(DUNE_COMMON_LIBS)            \
	$(DUNEMPILIBS)                                        \
  $(LDADD)


mpiremoteindicestest_SOURCES = mpiremoteindicestest.cc
mpiremoteindicestest_CPPFLAGS = $(AM_CPPFLAGS)  \
  $(DUNEMPICPPFLAGS)													  \
  $(DUNE_COMMON_CPPFLAGS)
mpiremoteindicestest_LDFLAGS = $(AM_LDFLAGS)    \
  $(DUNEMPILDFLAGS)                     	      \
  $(DUNE_COMMON_LDFLAGS)
mpiremoteindicestest_LDADD = 	                  \
  $(DUNE_COMMON_LDFLAGS) $(DUNE_COMMON_LIBS)	  \
  $(DUNEMPILIBS)                            	  \
	$(LDADD)


threadremoteindicestest_SOURCES = threadremoteindicestest.cc
threadremoteindicestest_CPPFLAGS = $(AM_CPPFLAGS)  \
	$(DUNE_COMMON_CPPFLAGS)
threadremoteindicestest_LDFLAGS = $(AM_LDFLAGS)    \
	$(DUNE_COMMON_LDFLAGS)
threadremoteindicestest_LDADD =                    \
	$(DUNE_COMMON_LDFLAGS) $(DUNE_COMMON_LIBS)	     \
	$(LDADD)																		     \
	-lpthread

include $(top_srcdir)/am/global-rules

EXTRA_DIST = CMakeLists.txt
