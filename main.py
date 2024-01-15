import sys
import socket
import logging
from mpi4py import MPI
import src.experiments

# set up logger
rank = MPI.COMM_WORLD.rank
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"%(asctime)s (%(created)f) | "
                              f"host={socket.gethostname().ljust(8, ' ')}:p{str(rank).zfill(2)} | "
                              f"%(name)s:%(funcName)s:%(levelname)s: %(message)s")
ch.setFormatter(formatter)
logging.root.handlers = []  # removing default logger
logging.root.addHandler(ch)  # adding my handler
logging.root.setLevel(logging.DEBUG)  # setting root logger level

def main():
    src.experiments.vascular_sprouting()

if __name__ == "__main__":
    main()