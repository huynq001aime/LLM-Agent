#!/bin/bash

python server1.py &
python server2.py &

sleep 5
python client.py