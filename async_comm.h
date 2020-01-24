#ifndef _ASYNC_COMM_H
#define _ASYNC_COMM_H

#include <mpi.h>
#include <vector>
#include <diy/serialization.hpp>
#include <stdint.h>

class AsyncComm { // not thread-safe. 
public:
  AsyncComm(MPI_Comm comm=MPI_COMM_WORLD, int max_nonblocking_requests=1024);
  ~AsyncComm();

  int rank() const {return _rank;}
  int np() const {return _np;}

  bool ready_to_send(); // ready to send
  bool isend(int dst_rank, int tag, diy::MemoryBuffer&);
  bool iprobe(int &src_rak, int tag, diy::MemoryBuffer&);

  uint64_t num_bytes_sent() const {return _num_bytes_sent;}

private:
  int get_nonblocking_request(); 
  void clear_finished_nonblocking_request(int num=-1);

private:
  const int _max_nonblocking_requests;
  std::vector<MPI_Request> _nonblocking_requests; 
  std::vector<diy::MemoryBuffer> _nonblocking_buffers; 
  std::set<int> _nonblocking_available, _nonblocking_occupied;

protected:
  int _np, _rank;
  MPI_Comm _comm_world;
  uint64_t _num_blocked_sends, _num_bytes_sent;
};

#endif
