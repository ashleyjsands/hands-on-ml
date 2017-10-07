source ../env/bin/activate
# This cluster has been designed to equally share the first GPU on the local machine.
python3 server.py 0 &
python3 server.py 1 &
python3 server.py 2 &
