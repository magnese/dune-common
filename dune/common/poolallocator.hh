// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
// $Id$
#ifndef DUNE_COMMON_POOLALLOCATOR_HH
#define DUNE_COMMON_POOLALLOCATOR_HH

#include "alignment.hh"
#include "static_assert.hh"
#include "lcm.hh"
#include <typeinfo>
#include <iostream>
#include <cassert>

template<std::size_t size, typename T>
int testPool();

//forward declarations.

namespace Dune
{

  template<typename T, std::size_t s>
  class Pool;

  template<typename T, std::size_t s>
  class PoolAllocator;

}

namespace std
{
  /*
     template<class T, std::size_t S>
     inline ostream& operator<<(ostream& os, Dune::Pool<T,S>& pool)
     {
     os<<"pool="<<&pool<<" allocated_="<<pool.allocated_;
     return os;
     }

     template<class T, std::size_t S>
     inline ostream& operator<<(ostream& os, Dune::PoolAllocator<T,S>& pool)
     {
     os<<pool.memoryPool_<<std::endl;
     return os;
     }
   */
}


namespace Dune
{
  /**
   * @file
   * This file implements the classes Pool and PoolAllocator providing
   * memory allocation for objects in chunks.
   * @author Markus Blatt
   */
  /**
   * @addtogroup Common
   *
   * @{
   */

  /**
   * @brief A memory pool of objects.
   *
   * The memory for the objects is organized in chunks.
   * Each chunks is capable of holding a specified number of
   * objects. The allocated objects will be properly aligned
   * for fast access.
   * Deallocated objects are cached for reuse to prevent memory
   * fragmentation.
   * @warning If the size of the objects allocated is less than the
   * size of a pointer memory is wasted.
   * @warning Due to aligned issues at the number of bytes of the
   * alignment prerequisite (< 4 bytes) are wasted. This effect
   * becomes negligable for big sizes of chunkSize.
   *
   * \tparam T The type that is allocated by us.
   * \tparam s The size of a memory chunk in bytes.
   */
  template<class T, std::size_t s>
  class Pool
  {
    friend int ::testPool<s,T>();

    //friend std::ostream& std::operator<<<>(std::ostream&,Pool<T,s>&);
    template< class, std::size_t > friend class PoolAllocator;

  private:

    /** @brief Reference to next free element. */
    struct Reference
    {
      Reference *next_;
    };

  public:

    /** @brief The type of object we allocate memory for. */
    typedef T MemberType;
    enum
    {

      /**
       * @brief The size of a union of Reference and MemberType.
       */
      unionSize = ((sizeof(MemberType) < sizeof(Reference)) ?
                   sizeof(Reference) : sizeof(MemberType)),

      /**
       * @brief Size requirement. At least one object has to
       * stored.
       */
      size = ((sizeof(MemberType) <= s && sizeof(Reference) <= s) ?
              s : unionSize),

      /**
       * @brief The alignment that suits both the MemberType and
       * the Reference (i.e. their least common multiple).
       */
      alignment = Lcm<AlignmentOf<MemberType>::value,AlignmentOf<Reference>::value>::value,

      /**
       * @brief The aligned size of the type.
       *
       * This size is bigger than sizeof of the type and a multiple of
       * the alignment requirement.
       */
      alignedSize = ((unionSize % alignment == 0) ?
                     unionSize :
                     ((unionSize / alignment + 1) * alignment)),

      /**
       * @brief The size of each chunk memory chunk.
       *
       * Will be adapted to be a multiple of the alignment plus
       * an offset to handle the case that the pointer to the memory
       * does not satisfy the alignment requirements.
       */
      chunkSize = ((size % alignment == 0) ?
                   size : ((size / alignment + 1)* alignment))
                  + alignment - 1,

      /**
       * @brief The number of element each chunk can hold.
       */
      elements = ((chunkSize - alignment + 1)/ alignedSize)
    };

  private:
    /** @brief Chunk of memory managed by the pool. */
    struct Chunk
    {

      //friend int testPool<s,T>();

      /** @brief The memory we hold. */
      char chunk_[chunkSize];

      /**
       * @brief Adress the first properly aligned
       * position in the chunk.
       */
      char* memory_;

      /** @brief The next element */
      Chunk *next_;

      /**
       * @brief Constructor.
       */
      Chunk()
      {
        // Make sure the alignment is correct!
        // long long should be 64bit safe!
        unsigned long long lmemory = reinterpret_cast<unsigned long long>(chunk_);
        if(lmemory % alignment != 0)
          lmemory = (lmemory / alignment + 1)
                    * alignment;

        memory_ = reinterpret_cast<char *>(lmemory);
      }
    };

  public:
    /** @brief Constructor. */
    inline Pool();
    /** @brief Destructor. */
    inline ~Pool();
    /**
     * @brief Get a new or recycled object
     * @return A pointer to the object memory.
     */
    inline T *allocate();
    /**
     * @brief Free an object.
     * @param o The pointer to memory block of the object.
     */
    inline void free(void* o);

    /**
     * @brief Print elements in pool for debugging.
     */
    inline void print(std::ostream& os);

  private:

    // Prevent Copying!
    Pool(const Pool<MemberType,s>&);

    void operator=(const Pool<MemberType,s>& pool) const;
    /** @brief Grow our pool.*/
    inline void grow();
    /** @brief The first free element. */
    Reference *head_;
    /** @brief Our memory chunks. */
    Chunk *chunks_;
    /* @brief The number of currently allocated elements. */
    //size_t allocated_;

  };

  /**
   * @brief An allocator managing a pool of objects for reuse.
   *
   * This allocator is specifically useful for small data types
   * where new and delete are too expensive.
   *
   * It uses a pool of memory chunks where the objects will be allocated.
   * This means that assuming that N objects fit into memory only every N-th
   * request for an object will result in memory allocation.
   *
   * @warning It is not suitable
   * for the use in standard containers as it cannot allocate
   * arrays of arbitrary size
   *
   * \tparam T The type that will be allocated.
   * \tparam s The number of elements to fit into one memory chunk.
   */
  template<class T, std::size_t s>
  class PoolAllocator
  {
    //friend std::ostream& std::operator<<<>(std::ostream&,PoolAllocator<T,s>&);

  public:
    /**
     * @brief Type of the values we construct and allocate.
     */
    typedef T value_type;

    enum
    {
      /**
       * @brief The number of objects to fit into one memory chunk
       * allocated.
       */
      size=s*sizeof(value_type)
    };

    /**
     * @brief The pointer type.
     */
    typedef T* pointer;

    /**
     * @brief The constant pointer type.
     */
    typedef const T* const_pointer;

    /**
     * @brief The reference type.
     */
    typedef T& reference;

    /**
     * @brief The constant reference type.
     */
    typedef const T& const_reference;

    /**
     * @brief The size type.
     */
    typedef std::size_t size_type;

    /**
     * @brief The difference_type.
     */
    typedef std::ptrdiff_t difference_type;

    /**
     * @brief Constructor.
     */
    inline PoolAllocator();

    /**
     * @brief Coopy Constructor.
     */
    template<typename U, std::size_t u>
    inline PoolAllocator(const PoolAllocator<U,u>&)
    {}

    /**
     * @brief Allocates objects.
     * @param n The number of objects to allocate. Has to be less
     * than Pool<T,s>::elements!
     * @param hint Ignored hint.
     * @return A pointer tp the allocated elements.
     */
    inline pointer allocate(size_t n, const_pointer hint=0);

    /**
     * @brief Free objects.
     *
     * Does not call the contructor!
     * @param n The number of object to free. Has to be one!
     * @param p Pointer to the first object.
     */
    inline void deallocate(pointer p, std::size_t n);

    /**
     * @brief Construct an object.
     * @param p Pointer to the object.
     * @param value The value to initialize it to.
     */
    inline void construct(pointer p, const_reference value);

    /**
     * @brief Destroy an object without freeing memory.
     * @param p Pointer to the object.
     */
    inline void destroy(pointer p);

    /**
     * @brief Convert a reference to a pointer.
     */
    inline pointer  address(reference x) const { return &x; }


    /**
     * @brief Convert a reference to a pointer.
     */
    inline const_pointer address(const_reference x) const { return &x; }

    /**
     * @brief Not correctly implemented, yet!
     */
    inline int max_size() const throw(){ return 1;}

    /**
     * @brief Rebind the allocator to another type.
     */
    template<class U>
    struct rebind
    {
      typedef PoolAllocator<U,s> other;
    };

    /** @brief The type of the memory pool we use. */
    typedef Pool<T,size> PoolType;

  private:
    /**
     * @brief The underlying memory pool.
     */
    static PoolType memoryPool_;
  };

  // specialization for void
  template <std::size_t s>
  class PoolAllocator<void,s>
  {
  public:
    typedef void*       pointer;
    typedef const void* const_pointer;
    // reference to void members are impossible.
    typedef void value_type;
    template <class U> struct rebind
    {
      typedef PoolAllocator<U,s> other;
    };

    template<typename T, std::size_t t>
    PoolAllocator(const PoolAllocator<T,t>&)
    {}

  };


  template<typename T1, std::size_t t1, typename T2, std::size_t t2>
  bool operator==(const PoolAllocator<T1,t1>&, const PoolAllocator<T2,t2>&)
  {
    return false;
  }


  template<typename T1, std::size_t t1, typename T2, std::size_t t2>
  bool operator!=(const PoolAllocator<T1,t1>&, const PoolAllocator<T2,t2>&)
  {
    return true;
  }

  template<typename T, std::size_t t1, std::size_t t2>
  bool operator==(const PoolAllocator<T,t1>&, const PoolAllocator<T,t2>&)
  {
    return Pool<T,t1>::chunkSize == Pool<T,t2>::chunkSize;
  }


  template<typename T, std::size_t t1, std::size_t t2>
  bool operator!=(const PoolAllocator<T,t1>&, const PoolAllocator<T,t2>&)
  {
    return Pool<T,t1>::chunkSize != Pool<T,t2>::chunkSize;
  }


  template<typename T, std::size_t t1, std::size_t t2>
  bool operator==(const PoolAllocator<T,t1>&, const PoolAllocator<void,t2>&)
  {
    return false;
  }


  template<typename T, std::size_t t1, std::size_t t2>
  bool operator!=(const PoolAllocator<T,t1>&, const PoolAllocator<void,t2>&)
  {
    return true;
  }

  template<typename T, std::size_t t1, std::size_t t2>
  bool operator==(const PoolAllocator<void,t1>&, const PoolAllocator<T,t2>&)
  {
    return false;
  }


  template<typename T, std::size_t t1, std::size_t t2>
  bool operator!=(const PoolAllocator<void,t1>&, const PoolAllocator<T,t2>&)
  {
    return true;
  }
  template<std::size_t t1, std::size_t t2>
  bool operator==(const PoolAllocator<void,t1>&, const PoolAllocator<void,t2>&)
  {
    return true;
  }

  template<std::size_t t1, std::size_t t2>
  bool operator!=(const PoolAllocator<void,t1>&, const PoolAllocator<void,t2>&)
  {
    return false;
  }

  template<class T, std::size_t S>
  inline Pool<T,S>::Pool()
    : head_(0), chunks_(0) //, allocated_(0)
  {
    dune_static_assert(sizeof(T)<=unionSize, "Library Error: type T is too big");
    dune_static_assert(sizeof(Reference)<=unionSize, "Library Error: type of referene is too big");
    dune_static_assert(unionSize<=alignedSize, "Library Error: alignedSize too small");
    dune_static_assert(sizeof(T)<=chunkSize, "Library Error: chunkSize must be able to hold at least one value");
    dune_static_assert(sizeof(Reference)<=chunkSize, "Library Error: chunkSize must be able to hold at least one reference");
    dune_static_assert((chunkSize - (alignment - 1)) % alignment == 0, "Library Error: compiler cannot calculate!");
    dune_static_assert(elements>=1, "Library Error: we need to hold at least one element!");
    dune_static_assert(elements*alignedSize<=chunkSize, "Library Error: aligned elements must fit into chuck!");
    /*std::cout<<"s= "<<S<<" : T: "<<sizeof(T)<<" Reference: "<<sizeof(Reference)<<" union: "<<unionSize<<" alignment: "<<alignment<<
       "aligned: "<<alignedSize<<" chunk: "<< chunkSize<<" elements: "<<elements<<std::endl;*/
  }

  template<class T, std::size_t S>
  inline Pool<T,S>::~Pool()
  {
    /*
       if(allocated_!=0)
       std::cerr<<"There are still "<<allocated_<<" allocated elements by the Pool<"<<typeid(T).name()<<","<<S<<"> "
               <<static_cast<void*>(this)<<"! This is a memory leak and might result in segfaults"
               <<std::endl;
     */
    // delete the allocated chunks.
    Chunk *current=chunks_;

    while(current!=0)
    {
      Chunk *tmp = current;
      current = current->next_;
      delete tmp;
    }
  }

  template<class T, std::size_t S>
  inline void Pool<T,S>::print(std::ostream& os)
  {
    Chunk* current=chunks_;
    while(current) {
      os<<current<<" ";
      current=current->next_;
    }
    os<<current<<" ";
  }

  template<class T, std::size_t S>
  inline void Pool<T,S>::grow()
  {
    Chunk *newChunk = new Chunk;
    newChunk->next_ = chunks_;
    chunks_ = newChunk;

    char* start = reinterpret_cast<char *>(chunks_->memory_);
    char* last  = &start[(elements-1)*alignedSize];

    for(char* element=start; element<last; element=element+alignedSize) {
      reinterpret_cast<Reference*>(element)->next_
        = reinterpret_cast<Reference*>(element+alignedSize);
    }

    reinterpret_cast<Reference*>(last)->next_=0;
    head_ = reinterpret_cast<Reference*>(start);
  }

  template<class T, std::size_t S>
  inline void Pool<T,S>::free(void* b)
  {
    if(b) {
      Reference* freed = reinterpret_cast<Reference*>(b);
      freed->next_ = head_;
      head_ = freed;
      //--allocated_;
    }else
      std::cerr<< "Tried to free null pointer! "<<b<<std::endl;
  }

  template<class T, std::size_t S>
  inline T* Pool<T,S>::allocate()
  {
    if(!head_)
      grow();

    Reference* p = head_;
    head_ = p->next_;
    //++allocated_;
    return reinterpret_cast<T*>(p);
  }

  template<class T, std::size_t s>
  typename PoolAllocator<T,s>::PoolType PoolAllocator<T,s>::memoryPool_;

  template<class T, std::size_t s>
  inline PoolAllocator<T,s>::PoolAllocator()
  { }

  template<class T, std::size_t s>
  inline T* PoolAllocator<T,s>::allocate(std::size_t n, const T* hint)
  {
    assert(n==1); //<=(Pool<T,s>::elements));
    return memoryPool_.allocate();
  }

  template<class T, std::size_t s>
  inline void PoolAllocator<T,s>::deallocate(T* p, std::size_t n)
  {
    for(size_t i=0; i<n; i++)
      memoryPool_.free(p++);
  }

  template<class T, std::size_t s>
  inline void PoolAllocator<T,s>::construct(T* p, const T& value)
  {
    ::new (static_cast<void*>(p))T(value);
  }

  template<class T, std::size_t s>
  inline void PoolAllocator<T,s>::destroy(T* p)
  {
    p->~T();
  }

  /** @} */
}
#endif