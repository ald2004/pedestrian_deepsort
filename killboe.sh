#!/bin/sh

ps -ef|grep flask_server |awk '{print $2}'|xargs kill -9
ps -ef|grep boe_merge_demo |awk '{print $2}'|xargs kill -9
