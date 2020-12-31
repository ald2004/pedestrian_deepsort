#!/bin/sh
nohup /usr/bin/python flask_server.py &
nohup /usr/bin/python boe_merge_demo.py x &
