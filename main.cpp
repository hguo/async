#include <diy/master.hpp>
#include <iostream>
#include <map>
#include <vector>
#include "async_comm.h"
#include "state_exchanger.h"
#include "concurrentqueue.h"

template <typename MsgType> 
// DTP is a minimalist distributed thread pool runtime
// requirement for MsgType: 
// - MsgType must have an integer blkid
// - MsgType must be serializable by diy
struct DTP {
  using MsgQueue = moodycamel::ConcurrentQueue<MsgType>;

  DTP(MPI_Comm comm = MPI_COMM_WORLD) : acomm(comm) {
    apply_default_block_assignment();
    qsend.resize(acomm.np());
    state.control(); // update state exchanger
  }

  void apply_default_block_assignment() { // use one block for each proc
    blk_proc_map.clear();
    proc_blk_map.clear();
    proc_blk_map.resize(acomm.np());
    for (auto i = 0; i < acomm.np(); i ++)
      assign_block_to_proc(i, i);
  }

  void assign_block_to_proc(int blkid, int rank) {
    blk_proc_map[blkid] = rank;
    proc_blk_map[rank].push_back(blkid);
  }

  virtual void f(/*const*/MsgType& msg) = 0; // message handler

  // if an inbound message is tiny, it will be directly consumed by the comm thread directly.
  virtual bool is_tiny_message(const MsgType& msg) const {return false;}

  virtual void enqueue(const MsgType& msg) {
    const int blkid = msg.blkid;
    if (blk_proc_map[blkid] == acomm.rank()) // message triaging
      qwork.enqueue(msg); // local work queue
    else 
      qsend[blk_proc_map[blkid]].enqueue(msg); // remote work queue
  }

  void exec() { // main loop
    std::vector<std::thread> workers; // worker threads
    for (auto i = 0; i < nthreads; i ++)
      workers.push_back(std::thread([=]() {exec_worker(i);}));

    exec_comm(); // the main thread is the comm thread

    for (auto i = 0; i < nthreads; i ++)
      workers[i].join();
  }

  void exec_worker(int tid) { // main loop for worker threads
    // fprintf(stderr, "rank=%d, thread=%d\n", acomm.rank(), tid);
    MsgType msg;
    while (!state.all_done()) {
      if (qwork.try_dequeue(msg)) {
        f(msg);
      }
    }
  }

  void exec_comm() { // mail loop for communication thread
    // fprintf(stderr, "rank=%d, comm thread\n", acomm.rank());
    while (!state.all_done()) {
      state.control(); // update state exchanger
      
      // outbound messages
      for (auto dstproc = 0; dstproc < acomm.np(); dstproc ++) {
        if (dstproc == acomm.rank()) continue;
        if (!acomm.ready_to_send()) continue;
       
        static MsgType msgs_[bulk_deque_size];
        size_t count = qsend[dstproc].try_dequeue_bulk(msgs_, bulk_deque_size);
        if (count) {
          std::vector<MsgType> msgs(msgs_, msgs_ + count);
          diy::MemoryBuffer bb;
          diy::save(bb, msgs);

          if (!acomm.isend(dstproc, 0, bb)) {
            // assert(false); // this rarely happens
            for (auto i = 0; i < msgs.size(); i ++) // in case isend fails, push the msg back to the send queue
              qsend[dstproc].enqueue(msgs[i]);
          }
        }
      }

      // inbound messages
      {
        int srcproc;
        diy::MemoryBuffer bb;
        while (acomm.iprobe(srcproc, 0, bb)) {
          std::vector<MsgType> msgs;
          diy::load(bb, msgs); 
          for (auto i = 0; i < msgs.size(); i ++) {
            if (is_tiny_message(msgs[i])) f(msgs[i]);
            else enqueue(msgs[i]);
          }
        }
      }
    }
  }

public:
  AsyncComm acomm;
  StateExchanger state;

  std::map<int, int> blk_proc_map;
  std::vector<std::vector<int>> proc_blk_map;

  MsgQueue qwork;
  std::vector<MsgQueue> qsend; // one queue for each proc.

  int nthreads = std::thread::hardware_concurrency() - 1;

public: // "hyper" parameters
  static const int bulk_deque_size = 2048;
};


//////

struct particle {
  float x[4] = {0}; // spatiotemporal coordinates.
  int nsteps = 0;
};

struct particle_tracer_message {
  int blkid;
  int hops = 0;
  std::vector<particle> unfinished_particles;
};

namespace diy {
  template <> struct Serialization<particle_tracer_message> {
    static void save(BinaryBuffer& bb, const particle_tracer_message& m) {
      diy::save(bb, m.blkid);
      diy::save(bb, m.unfinished_particles);
    }
    static void load(BinaryBuffer& bb, particle_tracer_message& m) {
      diy::load(bb, m.blkid);
      diy::load(bb, m.unfinished_particles);
    }
  };
}

struct distributed_particle_tracer : public DTP<particle_tracer_message> {
  bool is_tiny_message(const particle_tracer_message& m) const {
    return m.unfinished_particles.size() < 10;
  }

  void init() { // initialize seeds
    if (acomm.rank() == 0) { // kickoff message from rank 0
      const int nseeds = 1024;
      particle_tracer_message msg;
      for (auto i = 0; i < nseeds; i ++) {
        particle p;
        msg.unfinished_particles.push_back(p);
      }
      enqueue(msg);
      // state.add_work(nseeds);
      state.add_work(1); // nseeds);
      fprintf(stderr, "initialized.\n");
    }
  }

  void f(particle_tracer_message &msg) {
    if (msg.hops < 16) {
      msg.blkid = rand() % acomm.np(); // random walk
      msg.hops ++;
      fprintf(stderr, "hops=%d\n", msg.hops);
      enqueue(msg);
    } else {
      state.dec_work();
    }
  }
};

int main(int argc, char **argv)
{
  int required = MPI_THREAD_MULTIPLE, provided;
  MPI_Init_thread(&argc, &argv, required, &provided); 
  
  distributed_particle_tracer tracer;
  tracer.init();
  tracer.exec();

  MPI_Finalize();
  return 0;
}
