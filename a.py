# mpi_helloworld.py

from mpi4py import MPI


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
node_name = MPI.Get_processor_name() # get the name of the node

if rank == 0:
    print('I\'m process %d, and there are %d process total.' % (rank, size))
else:
    print('Ok, I\'m process %d' % rank)