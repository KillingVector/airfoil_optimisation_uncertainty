import numpy as np
from mpi4py import MPI


def main():

    # Communicator
    comm = MPI.COMM_WORLD

    if comm.rank == 0:

        # Data
        data = np.linspace(1, 23, 23)

        # Indices for each sub-task
        idx_arr = np.array_split(np.arange(np.shape(data)[0]), comm.size)
    else:
        data = None
        idx_arr = None

    # Broadcast data array to each process on comm
    data = comm.bcast(data, root=0)

    # Scatter index array to process on comm
    idx_arr = comm.scatter(idx_arr, root=0)

    # Process data (all processes on comm)
    out = foo(comm, data, idx_arr)

    # Gather output from all process on comm
    out = comm.gather(out, root=0)

    # Process output
    if comm.rank == 0:
        data_2 = np.zeros(0)
        for arr in out:
            data_2 = np.concatenate((data_2, arr))
        data_out = np.array([data, data_2])

        # Print output
        # print('out:', out)
        print('data_out:', data_out)


def foo(comm, data, idx_arr):

    print('comm.:', comm.rank)
    print('data:', data)
    print('idx_arr:', idx_arr)
    print('np.shape(idx_arr):', np.shape(idx_arr))
    print('------------------------------')

    out = np.zeros(np.shape(idx_arr)[0])
    for i, idx in enumerate(idx_arr):
        out[i] = data[idx]**2.0

    return out


if __name__ == '__main__':
    main()

