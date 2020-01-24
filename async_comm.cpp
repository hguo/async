#include "async_comm.h"
#include <list>
#include <cassert>
#include <limits.h>

static const int mpi_tag_shutdown = 3333; // TODO

AsyncComm::AsyncComm(MPI_Comm comm, int max_nonblocking_requests) :
  _comm_world(comm),
  _max_nonblocking_requests(max_nonblocking_requests),
  _nonblocking_requests(max_nonblocking_requests, MPI_REQUEST_NULL),
  _nonblocking_buffers(max_nonblocking_requests),
  _num_blocked_sends(0),
  _num_bytes_sent(0)
{
  MPI_Comm_size(comm, &_np);
  MPI_Comm_rank(comm, &_rank);

  for (int i=0; i<_max_nonblocking_requests; i++)
    _nonblocking_available.insert(i);
}

AsyncComm::~AsyncComm()
{
  // TODO
}

bool AsyncComm::ready_to_send()
{
  if (_nonblocking_available.size() > 0) return true;
  else {
    clear_finished_nonblocking_request(1);
    if (_nonblocking_available.size() > 0) return true;
    else return false;
  }
}

bool AsyncComm::isend(int dst_rank, int tag, diy::MemoryBuffer& bb)  
{
  if (bb.empty()) return false;

  int request = get_nonblocking_request(); 
  if (request == -1) {
    _num_blocked_sends ++; 
    return false; 
  }

  diy::MemoryBuffer &buffer = _nonblocking_buffers[request]; 
  buffer.swap(bb); 

  MPI_Isend(
    (void*)buffer.buffer.data(), 
    buffer.size(), 
    MPI_CHAR,
    dst_rank, 
    tag, 
    _comm_world, 
    &_nonblocking_requests[request]
  );

  // fprintf(stdout, "[rank=%d] sent %lu bytes\n", rank(), buffer.size());

  _num_bytes_sent += buffer.size();
  return true; 
}

bool AsyncComm::iprobe(int &src_rank, int tag, diy::MemoryBuffer& bb)
{
  MPI_Request request; 
  MPI_Status status; 
  int flag = 0; 

  MPI_Iprobe(
    MPI_ANY_SOURCE, 
    tag, 
    _comm_world, 
    &flag, 
    &status
  ); 

  if (flag) { // has message 
    int length; 

    src_rank = status.MPI_SOURCE;

    MPI_Get_count(&status, MPI_CHAR, &length); 
    bb.clear();
    bb.buffer.resize(length); 
    assert(length != 0);

    MPI_Irecv(
      (void*)bb.buffer.data(), 
      length, 
      MPI_CHAR, 
      status.MPI_SOURCE, 
      status.MPI_TAG, 
      _comm_world, 
      &request
    ); 
    MPI_Wait(&request, &status); 
  
    // fprintf(stdout, "[rank=%d] recv %lu bytes\n", rank(), bb.size());

    return true; 
  } else 
    return false; 
}


int AsyncComm::get_nonblocking_request()
{
  if (_nonblocking_available.empty()) 
    clear_finished_nonblocking_request(1); 
  
  if (_nonblocking_available.empty()) 
    return -1; // I do not want to try again

  std::set<int>::iterator it = _nonblocking_available.begin();
  const int i = *it;
  _nonblocking_available.erase(it);
  _nonblocking_occupied.insert(i);

  return i;
}

void AsyncComm::clear_finished_nonblocking_request(int num)
{
  if (num<0) num = INT_MAX;
  int count=0;

  std::list<std::set<int>::iterator> to_remove;
  for (std::set<int>::iterator it = _nonblocking_occupied.begin(); 
       it != _nonblocking_occupied.end(); 
       it ++) 
  {
    const int i = *it;
    if (_nonblocking_requests[i] == MPI_REQUEST_NULL) { // not likely to happen
      _nonblocking_buffers[i].clear();
      count ++; 
    } else {
      int flag; // flag of finish
      MPI_Test(
        &_nonblocking_requests[i], 
        &flag, 
        MPI_STATUS_IGNORE
      );

      if (flag) {
        MPI_Wait(
          &_nonblocking_requests[i], 
          MPI_STATUS_IGNORE
        );
        _nonblocking_buffers[i].clear();
        to_remove.push_back(it);
        _nonblocking_available.insert(i);
        count ++; 
      }
    }
    
    if (count >= num) break;
  }

  for (std::list<std::set<int>::iterator>::iterator it = to_remove.begin();
       it != to_remove.end();
       it ++)
  {
    _nonblocking_occupied.erase(*it);
  }
  // fprintf(stdout, "[rank=%d] cleared %d buffers\n", _rank, count);
}
