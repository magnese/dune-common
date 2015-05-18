// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_REMOTEINDICES_HH
#define DUNE_REMOTEINDICES_HH

#include "indexset.hh"
#include "plocalindex.hh"
#include "mpiparallelparadigm.hh"
#include "threadparallelparadigm.hh"
#include <dune/common/exceptions.hh>
#include <dune/common/poolallocator.hh>
#include <dune/common/sllist.hh>
#include <dune/common/stdstreams.hh>
#include <map>
#include <set>
#include <utility>
#include <iostream>
#include <algorithm>
#include <iterator>

namespace Dune
{
  /** @addtogroup Common_Parallel
   *
   * @{
   */
  /**
   * @file
   * @brief Classes describing a distributed indexset
   * @author Marco Agnese, Markus Blatt
   */

  template<typename T,typename P,typename A>
  class RemoteIndices;

  template<typename T1, typename T2>
  class RemoteIndex;

  template<typename T>
  class IndicesSyncer;

  template<typename T1, typename T2>
  std::ostream& operator<<(std::ostream& os, const RemoteIndex<T1,T2>& index);

  template<typename T, typename A, bool mode>
  class RemoteIndexListModifier;

  /** @brief Information about an index residing on another processor. */
  template<typename T1, typename T2>
  class RemoteIndex
  {
    template<typename T>
    friend class IndicesSyncer;

    template<typename T, typename A, typename R>
    friend void repairLocalIndexPointers(std::map<int,SLList<std::pair<typename T::GlobalIndex, typename T::LocalIndex::Attribute>,A> >&, R&, const T&);

    template<typename T, typename A, bool mode>
    friend class RemoteIndexListModifier;

  public:
    /** @brief the type of the global index. This type has to provide at least a operator&lt; for sorting. */
    typedef T1 GlobalIndex;
    /**
     * @brief The type of the attributes. Normally this will be an enumeration like
     * \code
     * enum Attributes{owner, border, overlap}
     * \endcode
     * e.g. OwnerOverlapCopyAttributes.
     */
    typedef T2 Attribute;

    /** @brief The type of the index pair. */
    typedef IndexPair<GlobalIndex,ParallelLocalIndex<Attribute> > PairType;

    /**
     * @brief Get the attribute of the index on the remote process.
     * @return The remote attribute.
     */
    const Attribute attribute() const;

    /**
     * @brief Get the corresponding local index pair.
     * @return The corresponding local index pair.
     */

    const PairType& localIndexPair() const;

    /** @brief Parameterless Constructor. */
    RemoteIndex();

    /**
     * @brief Constructor.
     * @param attribute The attribute of the index on the remote processor.
     * @param local The corresponding local index.
     */
    RemoteIndex(const T2& attribute, const PairType* local);

    /**
     * @brief Constructor.
     * Private as it should only be called from within Indexset.
     * @param attribute The attribute of the index on the remote processor.
     */
    RemoteIndex(const T2& attribute);

    bool operator==(const RemoteIndex& ri) const;

    bool operator!=(const RemoteIndex& ri) const;

  private:
    /** @brief The corresponding local index for this process. */
    const PairType* localIndex_;

    /** @brief The attribute of the index on the other process. */
    char attribute_;
  };

  template<class T,class P,class A>
  std::ostream& operator<<(std::ostream& os, const RemoteIndices<T,P,A>& indices);

  class InterfaceBuilder;

  template<class T, class A>
  class CollectiveIterator;

  template<class T>
  class IndicesSyncer;

  // forward declaration needed for friend declaration.
  template<typename T1, typename T2>
  class OwnerOverlapCopyCommunication;

  /**
   * @brief The indices present on remote processes.
   *
   * To set up communication between the set of processes active in
   * the communication every process needs to know which
   * indices are also known to other processes and which attributes
   * are attached to them on the remote side.
   *
   * This information is managed by this class. The information can either
   * be computed automatically calling rebuild (which requires information
   * to be sent in a ring) or set up by hand using the
   * RemoteIndexListModifiers returned by function getModifier(int).
   *
   * @tparam T The type of the parallel index set.
   * @tparam P The type of the parallel paradigm to use.
   * @tparam A The type of the allocator to use.
   */
  template<class T, class P=MPIParadigm, class A=std::allocator<RemoteIndex<typename T::GlobalIndex, typename T::LocalIndex::Attribute> > >
  class RemoteIndices
  {
  public:
    /** @brief The type of the index set we use, e.g. ParallelLocalIndexSet. */
    typedef T ParallelIndexSet;

    /** @brief The type of the parallel paradigm we use, e.g. MPIParadigm. */
    typedef P ParallelParadigm;

  private:
    typedef RemoteIndices<T,P,A> ThisType;
    friend class InterfaceBuilder;
    friend class IndicesSyncer<ThisType>;
    template<typename T1, typename A2, typename R>
    friend void repairLocalIndexPointers(std::map<int,SLList<std::pair<typename T1::GlobalIndex, typename T1::LocalIndex::Attribute>,A2> >&, R&, const T1&);

    template<class G, class T1, class T2>
    friend void fillIndexSetHoles(const G& graph, OwnerOverlapCopyCommunication<T1,T2>& oocomm);
    friend std::ostream& operator<<<>(std::ostream&, const RemoteIndices<T,P,A>&);

  public:
    /** @brief The type of the collective iterator over all remote indices. */
    typedef CollectiveIterator<ParallelIndexSet,A> CollectiveIteratorT;

    /** @brief The type of the global index. */
    typedef typename ParallelIndexSet::GlobalIndex GlobalIndex;

    /** @brief The type of the local index. */
    typedef typename ParallelIndexSet::LocalIndex LocalIndex;

    /** @brief The type of the attribute. */
    typedef typename LocalIndex::Attribute Attribute;

    /** @brief Type of the remote indices we manage. */
    typedef Dune::RemoteIndex<GlobalIndex,Attribute> RemoteIndex;

    /** @brief The type of the allocator for the remote index list. */
    typedef typename A::template rebind<RemoteIndex>::other Allocator;

    /** @brief The type of the remote index list. */
    typedef SLList<RemoteIndex,Allocator> RemoteIndexList;

    /** @brief The type of the map from rank to remote index list. */
    typedef std::map<int, std::pair<RemoteIndexList*,RemoteIndexList*> > RemoteIndexMap;

    typedef typename RemoteIndexMap::const_iterator const_iterator;

    /** @brief The type of the communicator. */
    typedef typename ParallelParadigm::CommType CommType;

    /**
     * @brief Constructor.
     * @param parallel The parallel paradigm to use.
     * @param source The indexset which represents the global to local mapping at the source of the communication.
     * @param destination The indexset to which the communication which represents the global to local mapping at the destination of the communication.
     * May be the same as the source indexset.
     * @param neighbours Optional: The neighbours the process shares indices with. If this parameter is omitted a ring communication with all indices will take
     * place to calculate this information which is O(P).
     * @param includeSelf If true, sending from indices of the processor to other indices on the same processor is enabled even if the same indexset is used
     * on both the sending and receiving side.
     */
    inline RemoteIndices(const ParallelIndexSet& source, const ParallelIndexSet& destination, ParallelParadigm& parallel,
                         const std::vector<int>& neighbours=std::vector<int>(), bool includeSelf=false);

    /** @brief Constructor. Available only for MPI. */
    inline RemoteIndices(const ParallelIndexSet& source, const ParallelIndexSet& destination, const typename MPIParadigm::CommType& comm,
                         const std::vector<int>& neighbours=std::vector<int>(), bool includeSelf=false);

    RemoteIndices();

    /**
     * @brief Tell whether sending from indices of the processor to other indices on the same processor is enabled even if the same indexset is
     * used on both the sending and receiving side.
     *
     * @param includeSelf If true it is enabled.
     */
    void setIncludeSelf(bool includeSelf);

    /**
     * @brief Set the index sets and the parallel paradigm we work with.
     *
     * @warning All remote indices already setup will be deleted!
     *
     * @param parallel The parallel paradigm to use.
     * @param source The indexset which represents the global to local mapping at the source of the communication.
     * @param destination The indexset to which the communication which represents the global to local mapping at the destination of the communication.
     * May be the same as the source indexset.
     * @param neighbours Optional: The neighbours the process shares indices with. If this parameter is omitted a ring communication with all indices will take
     * place to calculate this information which is O(P).
     */
    void setIndexSets(const ParallelIndexSet& source, const ParallelIndexSet& destination, ParallelParadigm& parallel,
                      const std::vector<int>& neighbours=std::vector<int>());


    /** @brief Set the index sets. Available only for MPI.*/
    void setIndexSets(const ParallelIndexSet& source, const ParallelIndexSet& destination, const typename MPIParadigm::CommType& comm,
                      const std::vector<int>& neighbours=std::vector<int>());


    template<typename C>
    void setNeighbours(const C& neighbours)
    {
      neighbourIds.clear();
      neighbourIds.insert(neighbours.begin(), neighbours.end());
    }

    const std::set<int>& getNeighbours() const
    {
      return neighbourIds;
    }

    /**  @brief Destructor. */
    ~RemoteIndices();

    /**
     * @brief Rebuilds the set of remote indices.
     *
     * This has to be called whenever the underlying index sets change.
     *
     * If the template parameter ignorePublic is true all indices will be treated as public.
     */
    template<bool ignorePublic>
    void rebuild();

    bool operator==(const RemoteIndices& ri);

    /**
     * @brief Checks whether the remote indices are synced with the indexsets.
     *
     * If they are not synced the remote indices need to be rebuild.
     * @return True if they are synced.
     */
    inline bool isSynced() const;

    inline CommType communicator() const;

    /** @brief Get the parallel paradigm used. */
    inline ParallelParadigm& parallelParadigm() const;

    /**
     * @brief Get a modifier for a remote index list.
     *
     * Sometimes the user knows in advance which indices will be present
     * on other processors, too. Then he can set them up using this modifier.
     *
     * @warning Use with care. If the remote index list is inconsistent
     * after the modification the communication might result in a dead lock!
     *
     * @tparam mode If true the index set corresponding to the remote indices might get modified.
     * Therefore the internal pointers to the indices need to be repaired.
     * @tparam send If true the remote index information at the sending side will
     * be modified, if false the receiving side.
     */
    template<bool mode, bool send>
    inline RemoteIndexListModifier<ParallelIndexSet,A,mode> getModifier(int process);

    /**
     * @brief Find an iterator over the remote index lists of a specific process.
     * @param proc The identifier of the process.
     * @return The iterator the remote index lists postioned at the process.
     * If theres is no list for this process, the end iterator is returned.
     */
    inline const_iterator find(int proc) const;

    /**
     * @brief Get an iterator over all remote index lists.
     * @return The iterator over all remote index lists postioned at the first process.
     */
    inline const_iterator begin() const;

    /**
     * @brief Get an iterator over all remote index lists.
     * @return The iterator over all remote index lists postioned at the end.
     */
    inline const_iterator end() const;

    /** @brief Get an iterator for colletively iterating over the remote indices of all remote processes. */
    template<bool send>
    inline CollectiveIteratorT iterator() const;

    /** @brief Free the index lists. */
    inline void free();

    /**
     * @brief Get the number of processors we share indices with.
     * @return The number of neighbours.
     */
    inline int neighbours() const;

    /** @brief Get the index set at the source. */
    inline const ParallelIndexSet& sourceIndexSet() const;

    /** @brief Get the index set at destination. */
    inline const ParallelIndexSet& destinationIndexSet() const;

  private:
    /** copying is forbidden. */
    RemoteIndices(const RemoteIndices&)
    {}

    /** @brief Index set used at the source of the communication. */
    const ParallelIndexSet* source_;

    /** @brief Index set used at the destination of the communication. */
    const ParallelIndexSet* target_;

    /** @brief Parallel paradigm used. */
    ParallelParadigm* parallel_;

    /** @brief Flag which indicates if parallel_ need to be free or not at decosntruction. */
    bool freeparallel_;

    /** @brief The neighbours we share indices with. If not empty this will speedup rebuild. */
    std::set<int> neighbourIds;

    /** @brief The sequence number of the source index set when the remote indices where build. */
    int sourceSeqNo_;

    /** @brief The sequence number of the destination index set when the remote indices where build. */
    int destSeqNo_;

    /** @brief Whether the public flag was ignored during the build. */
    bool publicIgnored;

    /** @brief Whether the next build will be the first build ever. */
    bool firstBuild;

    /*
     * @brief If true, sending from indices of the processor to other indices on the same processor is enabled even if
     * the same indexset is used on both the sending and receiving side.
     */
    bool includeSelf;

    /** @brief The index pair type. */
    typedef IndexPair<GlobalIndex,LocalIndex> PairType;

    /**
     * @brief The remote indices. The key is the process id and the values are the pair of remote index lists,
     * the first for receiving, the second for sending.
     */
    RemoteIndexMap remoteIndices_;

    /**
     * @brief Count the number of public indices in an index set.
     * @param indexSet The index set whose indices we count.
     * @return the number of indices marked as public.
     */
    inline int noPublic(const ParallelIndexSet& indexSet);
  };

  /** @} */

  /**
   * @brief Modifier for adding and/or deleting remote indices from
   * the remote index list.
   *
   * In some cases all the information about the indices also present
   * on remote process might already be known. In this case this
   * information can be provided to the RemoteIndices via this modifier.
   * This prevents the global communication needed by a call to
   * RemoteIndices::rebuild.
   *
   * In some cases it might advisable to run IndicesSyncer::sync afterwards.
   *
   * @warning Use with care. If the indices are not consistent afterwards
   * communication attempts might deadlock!
   */
  template<class T, class A, bool mode>
  class RemoteIndexListModifier
  {

    template<typename T1,typename P1,typename A1>
    friend class RemoteIndices;

  public:
    class InvalidPosition : public RangeError
    {};

    enum {
      /**
       * @brief If true the index set corresponding to the
       * remote indices might get modified.
       *
       * If for example new indices are added to an index set
       * all pointers of the remote indices to the local indices
       * become invalid after ParallelIndexSet::endResize() was called.
       */
      MODIFYINDEXSET=mode
    };

    /** @brief Type of the index set we use. */
    typedef T ParallelIndexSet;

    /** @brief The type of the global index. */
    typedef typename ParallelIndexSet::GlobalIndex GlobalIndex;

    /** @brief The type of the local index. */
    typedef typename ParallelIndexSet::LocalIndex LocalIndex;

    /** @brief The type of the attribute. */
    typedef typename LocalIndex::Attribute Attribute;

    /** @brief Type of the remote indices we manage. */
    typedef Dune::RemoteIndex<GlobalIndex,Attribute> RemoteIndex;

    /** @brief The type of the allocator for the remote index list. */
    typedef A Allocator;

    /** @brief The type of the remote index list. */
    typedef SLList<RemoteIndex,Allocator> RemoteIndexList;

    /** @brief The type of the modifying iterator of the remote index list. */
    typedef SLListModifyIterator<RemoteIndex,Allocator> ModifyIterator;

    /** @brief The type of the remote index list iterator. */
    typedef typename RemoteIndexList::const_iterator ConstIterator;

    /**
     * @brief Insert an index to the list.
     *
     * Moves to the position where the index fits and inserts it.
     * After the insertion only indices with an bigger global index
     * than the inserted can be inserted.
     *
     * This method is only available if MODIFYINDEXSET is false.
     *
     * @param index The index to insert.
     * @exception InvalidPosition Thrown if the index at the current position or
     * the one before has bigger global index than the one to be inserted.
     */
    void insert(const RemoteIndex& index) throw(InvalidPosition);

    /**
     * @brief Insert an index to the list.
     *
     * Moves to the position where the index fits and inserts it.
     * After the insertion only indices with an bigger global index
     * than the inserted can be inserted.
     *
     * This method is only available if MODIFYINDEXSET is true.
     *
     * @param index The index to insert.
     * @param global The global index of the remote index.
     * @exception InvalidPosition Thrown if the index at the current position or
     * the one before has bigger global index than the one to be inserted.
     */
    void insert(const RemoteIndex& index, const GlobalIndex& global) throw(InvalidPosition);

    /**
     * @brief Remove a remote index.
     * @param global The global index corresponding to the remote index.
     * @return True If there was a corresponding remote index.
     * @exception InvalidPostion If there was an insertion or deletion of
     * a remote index corresponding to a bigger global index before.
     */
    bool remove(const GlobalIndex& global) throw(InvalidPosition);

    /**
     * @brief Repair the pointers to the local index pairs.
     *
     * Due to adding new indices or/and deleting indices in the
     * index set all pointers to the local index pair might become
     * invalid during ParallelIndexSet::endResize().
     * This method repairs them.
     *
     * @exception InvalidIndexSetState Thrown if the underlying
     * index set is not in ParallelIndexSetState::GROUND mode (only when
     * compiled with DUNE_ISTL_WITH_CHECKING!).
     */
    void repairLocalIndexPointers() throw(InvalidIndexSetState);

    RemoteIndexListModifier(const RemoteIndexListModifier&);

    /**
     * @brief Default constructor.
     * @warning Object is not usable!
     */
    RemoteIndexListModifier() : glist_()
    {}

  private:
    /**
     * @brief Create a modifier for a remote index list.
     * @param indexSet The set of indices the process knows.
     * @param rList The list of remote indices to modify.
     */
    RemoteIndexListModifier(const ParallelIndexSet& indexSet, RemoteIndexList& rList);

    typedef SLList<GlobalIndex,Allocator> GlobalList;
    typedef typename GlobalList::ModifyIterator GlobalModifyIterator;
    RemoteIndexList* rList_;
    const ParallelIndexSet* indexSet_;
    GlobalList glist_;
    ModifyIterator iter_;
    GlobalModifyIterator giter_;
    ConstIterator end_;
    bool first_;
    GlobalIndex last_;
  };

  /** @brief A collective iterator for moving over the remote indices for all processes collectively. */
  template<class T, class A>
  class CollectiveIterator
  {
    /** @brief Type of the index set we use. */
    typedef T ParallelIndexSet;

    /** @brief The type of the global index. */
    typedef typename ParallelIndexSet::GlobalIndex GlobalIndex;

    /** @brief The type of the local index. */
    typedef typename ParallelIndexSet::LocalIndex LocalIndex;

    /** @brief The type of the attribute. */
    typedef typename LocalIndex::Attribute Attribute;

    /** @brief The remote index type */
    typedef Dune::RemoteIndex<GlobalIndex,Attribute> RemoteIndex;

    /** @brief The allocator of the remote indices. */
    typedef typename A::template rebind<RemoteIndex>::other Allocator;

    /** @brief The type of the remote index list. */
    typedef SLList<RemoteIndex,Allocator> RemoteIndexList;

    /** @brief The of map for storing the iterators. */
    typedef std::map<int,std::pair<typename RemoteIndexList::const_iterator, const typename RemoteIndexList::const_iterator> > Map;

  public:
    /** @brief The type of the map from rank to remote index list. */
    typedef std::map<int, std::pair<RemoteIndexList*,RemoteIndexList*> > RemoteIndexMap;

    /**
     * @brief Constructor.
     * @param map_ The map of the remote indices.
     * @param send True if we want iterate over the remote indices used for sending.
     */
    inline CollectiveIterator(const RemoteIndexMap& map_, bool send);

    /**
     * @brief Advances all underlying iterators.
     *
     * All iterators are advanced until they point to a remote index whose
     * global id is bigger or equal to global.
     * Iterators pointing to their end are removed.
     * @param global The index we search for.
     */
    inline void advance(const GlobalIndex& global);

    /**
     * @brief Advances all underlying iterators.
     *
     * All iterators are advanced until they point to a remote index whose
     * global id is bigger or equal to global.
     * Iterators pointing to their end are removed.
     * @param global The index we search for.
     * @param attribute The attribute we search for.
     */
    inline void advance(const GlobalIndex& global, const Attribute& attribute);

    CollectiveIterator& operator++();

    /** @brief Checks whether there are still iterators in the map. */
    inline bool empty();

    /**
     * @brief Iterator over the valid underlying iterators.
     *
     * An iterator is valid if it points to a remote index whose
     * global id is equal to the one currently examined.
     */
    class iterator
    {
    public:
      typedef typename Map::iterator RealIterator;
      typedef typename Map::iterator ConstRealIterator;


      //! \todo Please doc me!
      iterator(const RealIterator& iter, const ConstRealIterator& end, GlobalIndex& index) : iter_(iter), end_(end), index_(index), hasAttribute(false)
      {
        // move to the first valid entry
        while(iter_!=end_ && iter_->second.first->localIndexPair().global()!=index_)
          ++iter_;
      }

      iterator(const RealIterator& iter, const ConstRealIterator& end, GlobalIndex index, Attribute attribute) :
               iter_(iter), end_(end), index_(index), attribute_(attribute), hasAttribute(true)
      {
        // move to the first valid entry or the end
        while(iter_!=end_ && (iter_->second.first->localIndexPair().global()!=index_ || iter_->second.first->localIndexPair().local().attribute()!=attribute))
          ++iter_;
      }
      //! \todo Please doc me!
      iterator(const iterator& other) : iter_(other.iter_), end_(other.end_), index_(other.index_)
      {}

      //! \todo Please doc me!
      iterator& operator++()
      {
        ++iter_;
        // if entry is not valid move on
        while(iter_!=end_ && (iter_->second.first->localIndexPair().global()!=index_ ||
              (hasAttribute && iter_->second.first->localIndexPair().local().attribute()!=attribute_)))
          ++iter_;
        assert(iter_==end_ || (iter_->second.first->localIndexPair().global()==index_));
        assert(iter_==end_ || !hasAttribute || (iter_->second.first->localIndexPair().local().attribute()==attribute_));
        return *this;
      }

      //! \todo Please doc me!
      const RemoteIndex& operator*() const
      {
        return *(iter_->second.first);
      }

      //! \todo Please doc me!
      int process() const
      {
        return iter_->first;
      }

      //! \todo Please doc me!
      const RemoteIndex* operator->() const
      {
        return iter_->second.first.operator->();
      }

      //! \todo Please doc me!
      bool operator==(const iterator& other)
      {
        return other.iter_==iter_;
      }

      //! \todo Please doc me!
      bool operator!=(const iterator& other)
      {
        return other.iter_!=iter_;
      }

    private:
      iterator();

      RealIterator iter_;
      RealIterator end_;
      GlobalIndex index_;
      Attribute attribute_;
      bool hasAttribute;
    };

    iterator begin();

    iterator end();

  private:

    Map map_;
    GlobalIndex index_;
    Attribute attribute_;
    bool noattribute;
  };

  template<typename T1,typename T2>
  RemoteIndex<T1,T2>::RemoteIndex(const T2& attribute, const PairType* local) : localIndex_(local), attribute_(attribute)
  {}

  template<typename T1,typename T2>
  RemoteIndex<T1,T2>::RemoteIndex(const T2& attribute) : localIndex_(0), attribute_(attribute)
  {}

  template<typename T1,typename T2>
  RemoteIndex<T1,T2>::RemoteIndex() : localIndex_(0), attribute_()
  {}
  template<typename T1,typename T2>
  inline bool RemoteIndex<T1,T2>::operator==(const RemoteIndex& ri) const
  {
    return localIndex_==ri.localIndex_ && attribute_==ri.attribute;
  }

  template<typename T1,typename T2>
  inline bool RemoteIndex<T1,T2>::operator!=(const RemoteIndex& ri) const
  {
    return localIndex_!=ri.localIndex_ || attribute_!=ri.attribute_;
  }

  template<typename T1,typename T2>
  inline const T2 RemoteIndex<T1,T2>::attribute() const
  {
    return T2(attribute_);
  }

  template<typename T1,typename T2>
  inline const IndexPair<T1,ParallelLocalIndex<T2> >& RemoteIndex<T1,T2>::localIndexPair() const
  {
    return *localIndex_;
  }

  template<typename T,typename P,typename A>
  inline RemoteIndices<T,P,A>::RemoteIndices(const ParallelIndexSet& source, const ParallelIndexSet& destination, ParallelParadigm& parallel,
                                             const std::vector<int>& neighbours, bool includeSelf_) : source_(&source), target_(&destination),
                                             parallel_(&parallel), freeparallel_(false), sourceSeqNo_(-1), destSeqNo_(-1), publicIgnored(false),
                                             firstBuild(true), includeSelf(includeSelf_)
  {
    setNeighbours(neighbours);
  }

  template<typename T,typename P,typename A>
  inline RemoteIndices<T,P,A>::RemoteIndices(const ParallelIndexSet& source, const ParallelIndexSet& destination, const typename MPIParadigm::CommType& comm,
                                             const std::vector<int>& neighbours, bool includeSelf_) : source_(&source), target_(&destination), freeparallel_(true),
                                             sourceSeqNo_(-1), destSeqNo_(-1), publicIgnored(false), firstBuild(true), includeSelf(includeSelf_)
  {
    parallel_ = new ParallelParadigm(comm);
    setNeighbours(neighbours);
  }


  template<typename T,typename P,typename A>
  void RemoteIndices<T,P,A>::setIncludeSelf(bool b)
  {
    includeSelf=b;
  }

  template<typename T,typename P,typename A>
  RemoteIndices<T,P,A>::RemoteIndices() : source_(0), target_(0), sourceSeqNo_(-1), destSeqNo_(-1), publicIgnored(false), firstBuild(true)
  {}

  template<typename T,typename P,typename A>
  void RemoteIndices<T,P,A>::setIndexSets(const ParallelIndexSet& source, const ParallelIndexSet& destination, ParallelParadigm& parallel,
                                          const std::vector<int>& neighbours)
  {
    free();
    source_ = &source;
    target_ = &destination;
    parallel_= &parallel;
    firstBuild = true;
    setNeighbours(neighbours);
  }

  template<typename T,typename P,typename A>
  void RemoteIndices<T,P,A>::setIndexSets(const ParallelIndexSet& source, const ParallelIndexSet& destination, const typename MPIParadigm::CommType& comm,
                                          const std::vector<int>& neighbours)
  {
    free();
    if(freeparallel_)
      delete parallel_;
    source_ = &source;
    target_ = &destination;
    parallel_= new ParallelParadigm(comm);
    firstBuild = true;
    setNeighbours(neighbours);
  }

  template<typename T,typename P,typename A>
  const typename RemoteIndices<T,P,A>::ParallelIndexSet& RemoteIndices<T,P,A>::sourceIndexSet() const
  {
    return *source_;
  }

  template<typename T,typename P,typename A>
  const typename RemoteIndices<T,P,A>::ParallelIndexSet& RemoteIndices<T,P,A>::destinationIndexSet() const
  {
    return *target_;
  }

  template<typename T,typename P,typename A>
  RemoteIndices<T,P,A>::~RemoteIndices()
  {
    free();
    if(freeparallel_)
      delete parallel_;
  }

  template<typename T,typename P,typename A>
  inline int RemoteIndices<T,P,A>::noPublic(const ParallelIndexSet& indexSet)
  {
    typedef typename ParallelIndexSet::const_iterator const_iterator;

    int noPublic=0;

    const const_iterator end=indexSet.end();
    for(const_iterator index=indexSet.begin(); index!=end; ++index)
      if(index->local().isPublic())
        noPublic++;

    return noPublic;
  }

  template<typename T,typename P,typename A>
  inline void RemoteIndices<T,P,A>::free()
  {
    typedef typename RemoteIndexMap::iterator Iterator;
    Iterator lend = remoteIndices_.end();
    for(Iterator lists=remoteIndices_.begin(); lists != lend; ++lists)
    {
      if(lists->second.first==lists->second.second)
      {
        // there is only one remote index list
        delete lists->second.first;
      }
      else
      {
        delete lists->second.first;
        delete lists->second.second;
      }
    }
    remoteIndices_.clear();
    firstBuild=true;
  }

  template<typename T,typename P,typename A>
  inline int RemoteIndices<T,P,A>::neighbours() const
  {
    return remoteIndices_.size();
  }

  template<typename T,typename P,typename A>
  template<bool ignorePublic>
  inline void RemoteIndices<T,P,A>::rebuild()
  {
    // test wether a rebuild is needed
    if(firstBuild || ignorePublic!=publicIgnored || ! isSynced())
    {
      free();
      parallel_->template buildRemote<ignorePublic,ParallelIndexSet,RemoteIndexList>(source_,target_,remoteIndices_,neighbourIds,includeSelf);
      sourceSeqNo_ = source_->seqNo();
      destSeqNo_ = target_->seqNo();
      firstBuild=false;
      publicIgnored=ignorePublic;
    }
  }

  template<typename T,typename P,typename A>
  inline bool RemoteIndices<T,P,A>::isSynced() const
  {
    return sourceSeqNo_==source_->seqNo() && destSeqNo_ ==target_->seqNo();
  }

  template<typename T,typename P,typename A>
  template<bool mode,bool send>
  RemoteIndexListModifier<T,A,mode> RemoteIndices<T,P,A>::getModifier(int process)
  {
    // the user are on their own now therefore we assume they know what they are doing and just set the remote indices to synced status
    sourceSeqNo_ = source_->seqNo();
    destSeqNo_ = target_->seqNo();

    typename RemoteIndexMap::iterator found = remoteIndices_.find(process);

    if(found == remoteIndices_.end())
    {
      if(source_ != target_)
        found = remoteIndices_.insert(found, std::make_pair(process, std::make_pair(new RemoteIndexList(), new RemoteIndexList())));
      else
      {
        RemoteIndexList* rlist = new RemoteIndexList();
        found = remoteIndices_.insert(found, std::make_pair(process, std::make_pair(rlist, rlist)));
      }
    }

    firstBuild = false;

    if(send)
      return RemoteIndexListModifier<T,A,mode>(*source_, *(found->second.first));
    else
      return RemoteIndexListModifier<T,A,mode>(*target_, *(found->second.second));
  }

  template<typename T,typename P,typename A>
  inline typename RemoteIndices<T,P,A>::ParallelParadigm& RemoteIndices<T,P,A>::parallelParadigm() const
  {
    return *parallel_;
  }

  template<typename T,typename P,typename A>
  inline typename RemoteIndices<T,P,A>::const_iterator RemoteIndices<T,P,A>::find(int proc) const
  {
    return remoteIndices_.find(proc);
  }

  template<typename T,typename P,typename A>
  inline typename RemoteIndices<T,P,A>::const_iterator RemoteIndices<T,P,A>::begin() const
  {
    return remoteIndices_.begin();
  }

  template<typename T,typename P,typename A>
  inline typename RemoteIndices<T,P,A>::const_iterator RemoteIndices<T,P,A>::end() const
  {
    return remoteIndices_.end();
  }

  template<typename T,typename P,typename A>
  bool RemoteIndices<T,P,A>::operator==(const RemoteIndices& ri)
  {
    if(neighbours()!=ri.neighbours())
      return false;

    typedef typename std::map<int,std::pair<RemoteIndexList*,RemoteIndexList*> >::const_iterator const_iterator;
    const const_iterator rend = remoteIndices_.end();

    for(const_iterator rindex = remoteIndices_.begin(), rindex1=ri.remoteIndices_.begin(); rindex!=rend; ++rindex, ++rindex1)
    {
      if(rindex->first != rindex1->first)
        return false;
      if(*(rindex->second.first) != *(rindex1->second.first))
        return false;
      if(*(rindex->second.second) != *(rindex1->second.second))
        return false;
    }
    return true;
  }

  template<class T, class A, bool mode>
  RemoteIndexListModifier<T,A,mode>::RemoteIndexListModifier(const ParallelIndexSet& indexSet, RemoteIndexList& rList) : rList_(&rList),
                                                             indexSet_(&indexSet), iter_(rList.beginModify()), end_(rList.end()),
                                                             first_(true)
  {
    if(MODIFYINDEXSET)
    {
      assert(indexSet_);
      for(ConstIterator iter=iter_; iter != end_; ++iter)
        glist_.push_back(iter->localIndexPair().global());
      giter_ = glist_.beginModify();
    }
  }

  template<typename T, typename A, bool mode>
  RemoteIndexListModifier<T,A,mode>::RemoteIndexListModifier(const RemoteIndexListModifier<T,A,mode>& other) : rList_(other.rList_),
                                                             indexSet_(other.indexSet_), glist_(other.glist_), iter_(other.iter_),
                                                             giter_(other.giter_), end_(other.end_), first_(other.first_), last_(other.last_)
  {}

  template<typename T, typename A, bool mode>
  inline void RemoteIndexListModifier<T,A,mode>::repairLocalIndexPointers() throw(InvalidIndexSetState)
  {
    if(MODIFYINDEXSET)
    {
      // repair pointers to local index set
#ifdef DUNE_ISTL_WITH_CHECKING
      if(indexSet_->state()!=GROUND)
        DUNE_THROW(InvalidIndexSetState, "Index has to be in ground mode for repairing pointers to indices");
#endif
      typedef typename ParallelIndexSet::const_iterator IndexIterator;
      typedef typename GlobalList::const_iterator GlobalIterator;
      typedef typename RemoteIndexList::iterator Iterator;
      GlobalIterator giter = glist_.begin();
      IndexIterator index = indexSet_->begin();

      for(Iterator iter=rList_->begin(); iter != end_; ++iter)
      {
        while(index->global()<*giter)
        {
          ++index;
#ifdef DUNE_ISTL_WITH_CHECKING
          if(index == indexSet_->end())
            DUNE_THROW(InvalidPosition, "No such global index in set!");
#endif
        }

#ifdef DUNE_ISTL_WITH_CHECKING
        if(index->global() != *giter)
          DUNE_THROW(InvalidPosition, "No such global index in set!");
#endif
        iter->localIndex_ = &(*index);
      }
    }
  }

  template<typename T, typename A, bool mode>
  inline void RemoteIndexListModifier<T,A,mode>::insert(const RemoteIndex& index) throw(InvalidPosition)
  {
    static_assert(!mode,"Not allowed if the mode indicates that new indices might be added to the underlying index set. Use insert"
                        "(const RemoteIndex&, const GlobalIndex&) instead");

#ifdef DUNE_ISTL_WITH_CHECKING
    if(!first_ && index.localIndexPair().global()<last_)
      DUNE_THROW(InvalidPosition, "Modifcation of remote indices have to occur with ascending global index.");
#endif
    // move to the correct position
    while(iter_ != end_ && iter_->localIndexPair().global() < index.localIndexPair().global())
      ++iter_;

    // no duplicate entries allowed
    assert(iter_==end_ || iter_->localIndexPair().global() != index.localIndexPair().global());
    iter_.insert(index);
    last_ = index.localIndexPair().global();
    first_ = false;
  }

  template<typename T, typename A, bool mode>
  inline void RemoteIndexListModifier<T,A,mode>::insert(const RemoteIndex& index, const GlobalIndex& global) throw(InvalidPosition)
  {
    static_assert(mode,"Not allowed if the mode indicates that no new indices might be added to the underlying index set. Use insert"
                       "(const RemoteIndex&) instead");
#ifdef DUNE_ISTL_WITH_CHECKING
    if(!first_ && global<last_)
      DUNE_THROW(InvalidPosition, "Modification of remote indices have to occur with ascending global index.");
#endif
    // move to the correct position
    while(iter_ != end_ && *giter_ < global)
    {
      ++giter_;
      ++iter_;
    }

    // no duplicate entries allowed
    assert(iter_->localIndexPair().global() != global);
    iter_.insert(index);
    giter_.insert(global);

    last_ = global;
    first_ = false;
  }

  template<typename T, typename A, bool mode>
  bool RemoteIndexListModifier<T,A,mode>::remove(const GlobalIndex& global) throw(InvalidPosition)
  {
#ifdef DUNE_ISTL_WITH_CHECKING
    if(!first_ && global<last_)
      DUNE_THROW(InvalidPosition, "Modifcation of remote indices have to occur with ascending global index.");
#endif

    bool found= false;

    if(MODIFYINDEXSET)
    {
      // move to the correct position
      while(iter_!=end_ && *giter_< global)
      {
        ++giter_;
        ++iter_;
      }
      if(*giter_ == global)
      {
        giter_.remove();
        iter_.remove();
        found=true;
      }
    }
    else
    {
      while(iter_!=end_ && iter_->localIndexPair().global() < global)
        ++iter_;
      if(iter_->localIndexPair().global()==global)
      {
        iter_.remove();
        found = true;
      }
    }

    last_ = global;
    first_ = false;
    return found;
  }

  template<typename T,typename P,typename A>
  template<bool send>
  inline typename RemoteIndices<T,P,A>::CollectiveIteratorT RemoteIndices<T,P,A>::iterator() const
  {
    return CollectiveIteratorT(remoteIndices_, send);
  }

  template<typename T,typename P,typename A>
  inline typename RemoteIndices<T,P,A>::CommType RemoteIndices<T,P,A>::communicator() const
  {
    return parallel_->communicator();
  }

  template<typename T, typename A>
  CollectiveIterator<T,A>::CollectiveIterator(const RemoteIndexMap& pmap, bool send)
  {
    typedef typename RemoteIndexMap::const_iterator const_iterator;
    const const_iterator end=pmap.end();
    for(const_iterator process=pmap.begin(); process != end; ++process)
    {
      const RemoteIndexList* list = send ? process->second.first : process->second.second;
      typedef typename RemoteIndexList::const_iterator iterator;
      map_.insert(std::make_pair(process->first, std::pair<iterator, const iterator>(list->begin(), list->end())));
    }
  }

  template<typename T, typename A>
  inline void CollectiveIterator<T,A>::advance(const GlobalIndex& index)
  {
    typedef typename Map::iterator iterator;
    typedef typename Map::const_iterator const_iterator;
    const const_iterator end = map_.end();
    for(iterator iter = map_.begin(); iter != end;)
    {
      // step the iterator until we are >= index
      typename RemoteIndexList::const_iterator current = iter->second.first;
      typename RemoteIndexList::const_iterator rend = iter->second.second;
      RemoteIndex remoteIndex;
      if(current != rend)
        remoteIndex = *current;

      while(iter->second.first!=iter->second.second && iter->second.first->localIndexPair().global()<index)
        ++(iter->second.first);

      // erase from the map if there are no more entries.
      if(iter->second.first == iter->second.second)
        map_.erase(iter++);
      else
        ++iter;
    }
    index_=index;
    noattribute=true;
  }

  template<typename T, typename A>
  inline void CollectiveIterator<T,A>::advance(const GlobalIndex& index, const Attribute& attribute)
  {
    typedef typename Map::iterator iterator;
    typedef typename Map::const_iterator const_iterator;
    const const_iterator end = map_.end();
    for(iterator iter = map_.begin(); iter != end;)
    {
      // step the iterator until we are >= index
      typename RemoteIndexList::const_iterator current = iter->second.first;
      typename RemoteIndexList::const_iterator rend = iter->second.second;
      RemoteIndex remoteIndex;
      if(current != rend)
        remoteIndex = *current;

      // move to global index or bigger
      while(iter->second.first!=iter->second.second && iter->second.first->localIndexPair().global()<index)
        ++(iter->second.first);

      // move to attribute or bigger
      while(iter->second.first!=iter->second.second && iter->second.first->localIndexPair().global()==index
            && iter->second.first->localIndexPair().local().attribute()<attribute)
        ++(iter->second.first);

      // erase from the map if there are no more entries.
      if(iter->second.first == iter->second.second)
        map_.erase(iter++);
      else
        ++iter;
    }
    index_=index;
    attribute_=attribute;
    noattribute=false;
  }

  template<typename T, typename A>
  inline CollectiveIterator<T,A>& CollectiveIterator<T,A>::operator++()
  {
    typedef typename Map::iterator iterator;
    typedef typename Map::const_iterator const_iterator;
    const const_iterator end = map_.end();
    for(iterator iter = map_.begin(); iter != end;)
    {
      // step the iterator until we are >= index
      typename RemoteIndexList::const_iterator current = iter->second.first;
      typename RemoteIndexList::const_iterator rend = iter->second.second;

      // move all iterators pointing to the current global index to next value
      if(iter->second.first->localIndexPair().global()==index_ && (noattribute || iter->second.first->localIndexPair().local().attribute() == attribute_))
        ++(iter->second.first);

      // erase from the map if there are no more entries.
      if(iter->second.first == iter->second.second)
        map_.erase(iter++);
      else
        ++iter;
    }
    return *this;
  }

  template<typename T, typename A>
  inline bool CollectiveIterator<T,A>::empty()
  {
    return map_.empty();
  }

  template<typename T, typename A>
  inline typename CollectiveIterator<T,A>::iterator CollectiveIterator<T,A>::begin()
  {
    if(noattribute)
      return iterator(map_.begin(), map_.end(), index_);
    else
      return iterator(map_.begin(), map_.end(), index_, attribute_);
  }

  template<typename T, typename A>
  inline typename CollectiveIterator<T,A>::iterator CollectiveIterator<T,A>::end()
  {
    return iterator(map_.end(), map_.end(), index_);
  }

  template<typename TG, typename TA>
  inline std::ostream& operator<<(std::ostream& os, const RemoteIndex<TG,TA>& index)
  {
    os<<"[global="<<index.localIndexPair().global()<<", remote attribute="<<index.attribute()<<" local attribute="<<index.localIndexPair().local().attribute()<<"]";
    return os;
  }

  template<typename T,typename P,typename A>
  inline std::ostream& operator<<(std::ostream& os, const RemoteIndices<T,P,A>& indices)
  {
    typedef typename RemoteIndices<T,P,A>::RemoteIndexList RList;
    typedef typename std::map<int,std::pair<RList*,RList*> >::const_iterator const_iterator;
    const const_iterator rend = indices.remoteIndices_.end();
    for(const_iterator rindex = indices.remoteIndices_.begin(); rindex!=rend; ++rindex)
    {
      os<<"Process "<<rindex->first<<":";

      if(!rindex->second.first->empty())
      {
        os<<" send:";

        const typename RList::const_iterator send= rindex->second.first->end();

        for(typename RList::const_iterator index = rindex->second.first->begin(); index != send; ++index)
          os<<*index<<" ";
        os<<std::endl;
      }
      if(!rindex->second.second->empty())
      {
        os<<"Process "<<rindex->first<<": "<<"receive: ";

        const typename RList::const_iterator rend= rindex->second.second->end();

        for(typename RList::const_iterator index = rindex->second.second->begin(); index != rend; ++index)
          os<<*index<<" ";
      }
      os<<std::endl<<std::flush;
    }
    return os;
  }
  /** @} */
}

#endif
