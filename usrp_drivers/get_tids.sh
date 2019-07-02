#!/bin/bash
for i in $(pgrep $1); do ps -mo pid,tid,fname,user,psr -p $i;done
