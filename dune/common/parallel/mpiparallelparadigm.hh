// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_MPIPARALLELPARADIGM_HH
#define DUNE_MPIPARALLELPARADIGM_HH

#include "indexset.hh"
#include "plocalindex.hh"
#include <dune/common/stdstreams.hh>
#include <map>
#include <set>
#include <iostream>

#if HAVE_MPI
#include "mpitraits.hh"
#include <mpi.h>

namespace Dune {
  /** @addtogroup Common_Parallel
   *
   * @{
   */
  /**
   * @file
   * @brief Class implementing the MPI parallel paradigm.
   * @author Marco Agnese, Markus Blatt
   */

  //! \todo Please doc me.
  template<typename TG,typename TA>
  class MPITraits<IndexPair<TG,ParallelLocalIndex<TA> > >
  {
  public:
    inline static MPI_Datatype getType();
  private:
    static MPI_Datatype type;
  };

  /** @brief MPIParadigm. */
  class MPIParadigm
  {

  public:
    /** @brief The type of the communicator. */
    typedef MPI_Comm CommType;

    /** @brief Constructor. */
    inline MPIParadigm(const CommType& comm);

    /** @brief Default constructor. */
    MPIParadigm()
    {}

    /** @brief Set the paradigm we work with. */
    inline void setParadigm(const CommType& comm);

    /** @brief Destructor. */
    ~MPIParadigm()
    {}

    /** @brief Get the mpi communicator used. */
    inline CommType communicator() const;

    //! \todo Please finsih to doc me.
    /**
     * @brief Build the remote mapping. If the template parameter ignorePublic is true all indices will be treated as public.
     * @param includeSelf If true, sending from indices of the processor to other indices on the same processor is enabled even
     * if the same indexset is used on both the sending and receiving side.
     */
    template<bool ignorePublic,class ParallelIndexSet, class RemoteIndexList, class RemoteIndexMap = std::map<int, std::pair<RemoteIndexList*,RemoteIndexList*> > >
    inline void buildRemote(const ParallelIndexSet* source, const ParallelIndexSet* target, RemoteIndexMap& remoteIndices, std::set<int>& neighbourIds, bool includeSelf);

  private:
    /** copying is forbidden. */
    MPIParadigm(const MPIParadigm&)
    {}

    /** @brief The communicator to use. */
    CommType comm_;

    /** @brief The communicator tag to use. */
    const static int commTag_=333;

    /**
     * @brief Pack the indices to send if source_ and target_ are the same.
     *
     * If the template parameter ignorePublic is true all indices will be treated
     * as public.
     * @param myPairs Array to store references to the public indices in.
     * @param p_out The output buffer to pack the entries to.
     * @param type The mpi datatype for the pairs.
     * @param bufferSize The size of the output buffer p_out.
     * @param position The position to start packing.
     */
    template<bool ignorePublic,class ParallelIndexSet>
    inline void packEntries(IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** myPairs, const ParallelIndexSet& indexSet, char* p_out,
                            MPI_Datatype type, int bufferSize, int* position, int n);

    /**
     * @brief unpacks the received indices and builds the remote index list.
     *
     * @param remote The list to add the indices to.
     * @param remoteEntries The number of remote entries to unpack.
     * @param local The local indices to check wether we know the remote indices.
     * @param localEntries The number of local indices.
     * @param type The mpi data type for unpacking.
     * @param p_in The input buffer to unpack from.
     * @param postion The position in the buffer to start unpacking from.
     * @param bufferSize The size of the input buffer.
     */
    template<class ParallelIndexSet,class RemoteIndexList>
    inline void unpackIndices(RemoteIndexList& remote, int remoteEntries, IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** local,
                              int localEntries, char* p_in, MPI_Datatype type, int* positon, int bufferSize, bool fromOurself);

    //! \todo Please doc me.
    template<class ParallelIndexSet,class RemoteIndexList>
    inline void unpackIndices(RemoteIndexList& send, RemoteIndexList& receive, int remoteEntries, IndexPair<typename ParallelIndexSet::GlobalIndex,
                              typename ParallelIndexSet::LocalIndex>** localSource, int localSourceEntries, IndexPair<typename ParallelIndexSet::GlobalIndex,
                              typename ParallelIndexSet::LocalIndex>** localDest, int localDestEntries, char* p_in, MPI_Datatype type, int* position, int bufferSize);

    //! \todo Please doc me.
    template<class ParallelIndexSet,class RemoteIndexList,class RemoteIndexMap>
    void unpackCreateRemote(char* p_in, IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** sourcePairs,
                            IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** DestPairs, RemoteIndexMap& remoteIndices,
                            int remoteProc, int sourcePublish, int destPublish, int bufferSize, bool sendTwo, bool fromOurSelf=false);
  };

  /** @} */

  template<typename TG,typename TA>
  MPI_Datatype MPITraits<IndexPair<TG,ParallelLocalIndex<TA> > >::getType()
  {
    if(type==MPI_DATATYPE_NULL) {
      int length[4];
      MPI_Aint disp[4];
      MPI_Datatype types[4] = {MPI_LB, MPITraits<TG>::getType(), MPITraits<ParallelLocalIndex<TA> >::getType(), MPI_UB};
      IndexPair<TG,ParallelLocalIndex<TA> > rep[2];
      length[0]=length[1]=length[2]=length[3]=1;
      MPI_Address(rep, disp); // lower bound of the datatype
      MPI_Address(&(rep[0].global_), disp+1);
      MPI_Address(&(rep[0].local_), disp+2);
      MPI_Address(rep+1, disp+3); // upper bound of the datatype
      for(int i=3; i >= 0; --i) disp[i] -= disp[0];
      MPI_Type_struct(4, length, disp, types, &type);
      MPI_Type_commit(&type);
    }
    return type;
  }

  template<typename TG,typename TA>
  MPI_Datatype MPITraits<IndexPair<TG,ParallelLocalIndex<TA> > >::type=MPI_DATATYPE_NULL;

  inline MPIParadigm::MPIParadigm(const CommType& comm) : comm_(comm)
  {}

  inline void MPIParadigm::setParadigm(const CommType& comm)
  {
    comm_ = comm;
  }

  inline typename MPIParadigm::CommType MPIParadigm::communicator() const
  {
    return comm_;
  }

  template<bool ignorePublic,typename ParallelIndexSet>
  inline void MPIParadigm::packEntries(IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** pairs, const ParallelIndexSet& indexSet, char* p_out,
                                      MPI_Datatype type, int bufferSize, int *position, int n)
  {
    // fill with own indices
    typedef typename ParallelIndexSet::const_iterator const_iterator;
    typedef IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex> PairType;
    const const_iterator end = indexSet.end();

    // pack the source indices
    int i=0;
    for(const_iterator index = indexSet.begin(); index != end; ++index)
      if(ignorePublic || index->local().isPublic()) {

        MPI_Pack(const_cast<PairType*>(&(*index)), 1, type, p_out, bufferSize, position, comm_);
        pairs[i++] = const_cast<PairType*>(&(*index));

      }
    assert(i==n);

  }

  template<typename ParallelIndexSet,typename RemoteIndexList,typename RemoteIndexMap>
  inline void MPIParadigm::unpackCreateRemote(char* p_in, IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** sourcePairs,
                                              IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** destPairs,
                                              RemoteIndexMap& remoteIndices, int remoteProc, int sourcePublish, int destPublish, int bufferSize, bool sendTwo, bool fromOurSelf)
  {
    typedef IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex> PairType;
    // unpack the number of indices we received
    int noRemoteSource=-1, noRemoteDest=-1;
    char twoIndexSets=0;
    int position=0;
    // did we receive two index sets?
    MPI_Unpack(p_in, bufferSize, &position, &twoIndexSets, 1, MPI_CHAR, comm_);
    // the number of source indices received
    MPI_Unpack(p_in, bufferSize, &position, &noRemoteSource, 1, MPI_INT, comm_);
    // the number of destination indices received
    MPI_Unpack(p_in, bufferSize, &position, &noRemoteDest, 1, MPI_INT, comm_);

    // indices for which we receive
    RemoteIndexList* receive= new RemoteIndexList();
    // indices for which we send
    RemoteIndexList* send=0;

    MPI_Datatype type= MPITraits<PairType>::getType();

    if(!twoIndexSets) {
      if(sendTwo) {
        send = new RemoteIndexList();
        // create both remote index sets simultaneously
        unpackIndices<ParallelIndexSet,RemoteIndexList>(*send, *receive, noRemoteSource, sourcePairs, sourcePublish, destPairs, destPublish, p_in, type, &position, bufferSize);
      }else{
        // we only need one list
        unpackIndices<ParallelIndexSet,RemoteIndexList>(*receive, noRemoteSource, sourcePairs, sourcePublish, p_in, type, &position, bufferSize, fromOurSelf);
        send=receive;
      }
    }else{

      int oldPos=position;
      // two index sets received
      unpackIndices<ParallelIndexSet,RemoteIndexList>(*receive, noRemoteSource, destPairs, destPublish, p_in, type, &position, bufferSize, fromOurSelf);
      if(!sendTwo)
        // unpack source entries again as destination entries
        position=oldPos;

      send = new RemoteIndexList();
      unpackIndices<ParallelIndexSet,RemoteIndexList>(*send, noRemoteDest, sourcePairs, sourcePublish, p_in, type, &position, bufferSize, fromOurSelf);
    }

    if(receive->empty() && send->empty()) {
      if(send==receive) {
        delete send;
      }else{
        delete send;
        delete receive;
      }
    }else{
      remoteIndices.insert(std::make_pair(remoteProc, std::make_pair(send,receive)));
    }

  }

  template<bool ignorePublic,typename ParallelIndexSet,typename RemoteIndexList,typename RemoteIndexMap>
  inline void MPIParadigm::buildRemote(const ParallelIndexSet* source, const ParallelIndexSet* target, RemoteIndexMap& remoteIndices, std::set<int>& neighbourIds, bool includeSelf)
  {
    // processor configuration
    int rank, procs;
    MPI_Comm_rank(comm_, &rank);
    MPI_Comm_size(comm_, &procs);

    // number of local indices to publish
    // the indices of the destination will be send
    int sourcePublish, destPublish;

    // do we need to send two index sets?
    char sendTwo = (source != target);

    if(procs==1 && !(sendTwo || includeSelf))
      // nothing to communicate
      return;

    typedef typename ParallelIndexSet::const_iterator const_iterator;

    int noPublicSource=0;
    const const_iterator sourceItEnd=source->end();
    for(const_iterator it=source->begin(); it!=sourceItEnd; ++it)
      if(it->local().isPublic())
        ++noPublicSource;

    sourcePublish = (ignorePublic) ? source->size() : noPublicSource;

    if(sendTwo)
    {
      int noPublicTarget=0;
      const const_iterator targetItEnd=target->end();
        for(const_iterator it=target->begin(); it!=targetItEnd; ++it)
          if(it->local().isPublic())
            ++noPublicTarget;

      destPublish = (ignorePublic) ? target->size() : noPublicTarget;
    }
    else
      // we only need to send one set of indices
      destPublish = 0;

    int maxPublish, publish=sourcePublish+destPublish;

    // calculate maximum number of indices send
    MPI_Allreduce(&publish, &maxPublish, 1, MPI_INT, MPI_MAX, comm_);

    // allocate buffers
    typedef IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex> PairType;

    PairType** destPairs;
    PairType** sourcePairs = new PairType*[sourcePublish>0 ? sourcePublish : 1];

    if(sendTwo)
      destPairs = new PairType*[destPublish>0 ? destPublish : 1];
    else
      destPairs=sourcePairs;

    char** buffer = new char*[2];
    int bufferSize;
    int position=0;
    int intSize;
    int charSize;

    // calculate buffer size
    MPI_Datatype type = MPITraits<PairType>::getType();

    MPI_Pack_size(maxPublish, type, comm_, &bufferSize);
    MPI_Pack_size(1, MPI_INT, comm_, &intSize);
    MPI_Pack_size(1, MPI_CHAR, comm_, &charSize);
    // our message will contain the following:
    // a bool wether two index sets where sent,
    // the size of the source and the dest indexset,
    // then the source and destination indices
    bufferSize += 2 * intSize + charSize;

    if(bufferSize<=0) bufferSize=1;

    buffer[0] = new char[bufferSize];
    buffer[1] = new char[bufferSize];

    // pack entries into buffer[0], p_out below
    MPI_Pack(&sendTwo, 1, MPI_CHAR, buffer[0], bufferSize, &position, comm_);

    // the number of indices we send for each index set
    MPI_Pack(&sourcePublish, 1, MPI_INT, buffer[0], bufferSize, &position, comm_);
    MPI_Pack(&destPublish, 1, MPI_INT, buffer[0], bufferSize, &position, comm_);

    // now pack the source indices and setup the destination pairs
    packEntries<ignorePublic,ParallelIndexSet>(sourcePairs, *source, buffer[0], type, bufferSize, &position, sourcePublish);
    // if necessary send the dest indices and setup the source pairs
    if(sendTwo)
      packEntries<ignorePublic,ParallelIndexSet>(destPairs, *target, buffer[0], type, bufferSize, &position, destPublish);

    // update remote indices for ourself
    if(sendTwo|| includeSelf)
      unpackCreateRemote<ParallelIndexSet,RemoteIndexList,RemoteIndexMap>(buffer[0], sourcePairs, destPairs, remoteIndices, rank, sourcePublish, destPublish, bufferSize,
                                                                          sendTwo, includeSelf);

    neighbourIds.erase(rank);

    if(neighbourIds.size()==0)
    {
      dvverb<<rank<<": Sending messages in a ring"<<std::endl;
      // send messages in ring
      for(int proc=1; proc<procs; proc++) {
        // pointers to the current input and output buffers
        char* p_out = buffer[1-(proc%2)];
        char* p_in = buffer[proc%2];

        MPI_Status status;
        if(rank%2==0) {
          MPI_Ssend(p_out, bufferSize, MPI_PACKED, (rank+1)%procs, commTag_, comm_);
          MPI_Recv(p_in, bufferSize, MPI_PACKED, (rank+procs-1)%procs, commTag_, comm_, &status);
        }else{
          MPI_Recv(p_in, bufferSize, MPI_PACKED, (rank+procs-1)%procs, commTag_, comm_, &status);
          MPI_Ssend(p_out, bufferSize, MPI_PACKED, (rank+1)%procs, commTag_, comm_);
        }

        // the process these indices are from
        int remoteProc = (rank+procs-proc)%procs;

        unpackCreateRemote<ParallelIndexSet,RemoteIndexList,RemoteIndexMap>(p_in, sourcePairs, destPairs, remoteIndices, remoteProc, sourcePublish,
                                                                            destPublish, bufferSize, sendTwo);

      }

    }
    else
    {
      MPI_Request* requests=new MPI_Request[neighbourIds.size()];
      MPI_Request* req=requests;

      typedef typename std::set<int>::size_type size_type;
      size_type noNeighbours=neighbourIds.size();

      // setup sends
      for(std::set<int>::iterator neighbour=neighbourIds.begin();
          neighbour!= neighbourIds.end(); ++neighbour) {
        // only send the information to the neighbouring processors
        MPI_Issend(buffer[0], position , MPI_PACKED, *neighbour, commTag_, comm_, req++);
      }

      // test for received messages
      for(size_type received=0; received <noNeighbours; ++received)
      {
        MPI_Status status;
        // probe for next message
        MPI_Probe(MPI_ANY_SOURCE, commTag_, comm_, &status);
        int remoteProc=status.MPI_SOURCE;
        int size;
        MPI_Get_count(&status, MPI_PACKED, &size);
        // receive message
        MPI_Recv(buffer[1], size, MPI_PACKED, remoteProc, commTag_, comm_, &status);

        unpackCreateRemote<ParallelIndexSet,RemoteIndexList,RemoteIndexMap>(buffer[1], sourcePairs, destPairs, remoteIndices, remoteProc, sourcePublish,
                                                                            destPublish, bufferSize, sendTwo);
      }
      // wait for completion of pending requests
      MPI_Status* statuses = new MPI_Status[neighbourIds.size()];

      if(MPI_ERR_IN_STATUS==MPI_Waitall(neighbourIds.size(), requests, statuses)) {
        for(size_type i=0; i < neighbourIds.size(); ++i)
          if(statuses[i].MPI_ERROR!=MPI_SUCCESS) {
            std::cerr<<rank<<": MPI_Error occurred while receiving message."<<std::endl;
            MPI_Abort(comm_, 999);
          }
      }
      delete[] requests;
      delete[] statuses;
    }


    // delete allocated memory
    if(destPairs!=sourcePairs)
      delete[] destPairs;

    delete[] sourcePairs;
    delete[] buffer[0];
    delete[] buffer[1];
    delete[] buffer;

  }

  template<typename ParallelIndexSet,typename RemoteIndexList>
  inline void MPIParadigm::unpackIndices(RemoteIndexList& remote, int remoteEntries, IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** local,
                                        int localEntries, char* p_in, MPI_Datatype type, int* position, int bufferSize, bool fromOurSelf)
  {

    typedef typename RemoteIndexList::MemberType RemoteIndex;
    typedef typename ParallelIndexSet::GlobalIndex GlobalIndex;
    typedef typename ParallelIndexSet::LocalIndex LocalIndex;
    typedef IndexPair<GlobalIndex,LocalIndex> PairType;

    if(remoteEntries==0)
      return;

    PairType index(-1);
    MPI_Unpack(p_in, bufferSize, position, &index, 1, type, comm_);
    GlobalIndex oldGlobal=index.global();
    int n_in=0, localIndex=0;

    // check if we know the global index
    while(localIndex<localEntries) {
      if(local[localIndex]->global()==index.global()) {
        int oldLocalIndex=localIndex;

        while(localIndex<localEntries && local[localIndex]->global()==index.global()) {
          if(!fromOurSelf || index.local().attribute() != local[localIndex]->local().attribute())
            // if index is from us it has to have a different attribute
            remote.push_back(RemoteIndex(index.local().attribute(), local[localIndex]));
          localIndex++;
        }

        // unpack next remote index
        if((++n_in) < remoteEntries) {
          MPI_Unpack(p_in, bufferSize, position, &index, 1, type, comm_);
          if(index.global()==oldGlobal)
            // restart comparison for the same global indices
            localIndex=oldLocalIndex;
          else
            oldGlobal=index.global();
        }else{
          // no more received indices
          break;
        }
        continue;
      }

      if (local[localIndex]->global()<index.global()) {
        // compare with next entry in our list
        ++localIndex;
      }else{
        // we do not know the index, unpack next
        if((++n_in) < remoteEntries) {
          MPI_Unpack(p_in, bufferSize, position, &index, 1, type, comm_);
          oldGlobal=index.global();
        }else
          // no more received indices
          break;
      }
    }

    // unpack the other received indices without doing anything
    while(++n_in < remoteEntries)
      MPI_Unpack(p_in, bufferSize, position, &index, 1, type, comm_);

  }

  template<typename ParallelIndexSet,typename RemoteIndexList>
  inline void MPIParadigm::unpackIndices(RemoteIndexList& send, RemoteIndexList& receive, int remoteEntries,
                                        IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** localSource, int localSourceEntries,
                                        IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex>** localDest, int localDestEntries,
                                        char* p_in, MPI_Datatype type, int* position, int bufferSize)
  {

    typedef IndexPair<typename ParallelIndexSet::GlobalIndex,typename ParallelIndexSet::LocalIndex> PairType;
    typedef typename RemoteIndexList::MemberType RemoteIndex;
    int n_in=0, sourceIndex=0, destIndex=0;

    // check if we know the global index
    while(n_in<remoteEntries && (sourceIndex<localSourceEntries || destIndex<localDestEntries)) {
      // unpack next index
      PairType index;
      MPI_Unpack(p_in, bufferSize, position, &index, 1, type, comm_);
      n_in++;

      // advance until global index in localSource and localDest are >= than the one in the unpacked index
      while(sourceIndex<localSourceEntries && localSource[sourceIndex]->global()<index.global())
        sourceIndex++;

      while(destIndex<localDestEntries && localDest[destIndex]->global()<index.global())
        destIndex++;

      // add a remote index if we found the global index
      if(sourceIndex<localSourceEntries && localSource[sourceIndex]->global()==index.global())
        send.push_back(RemoteIndex(index.local().attribute(), localSource[sourceIndex]));

      if(destIndex < localDestEntries && localDest[destIndex]->global() == index.global())
        receive.push_back(RemoteIndex(index.local().attribute(), localDest[sourceIndex]));
    }

  }

  /** @} */

}
#else
/* @brief Empty class needed when MPI is not defined for default template parameter in RemoteIndices. */
class MPIParadigm{};
#endif
#endif
