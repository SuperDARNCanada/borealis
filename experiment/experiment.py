#!/usr/bin/python

# write an experiment that creates a new control program.

from controlprog import controlprog

def experiment():
	prog=controlprog(1,150)
	prog.cpo_list[0].freq=13124
	prog.cpo_list[0].sequence=[0,5,8,12]
	prog.cpo_list[0].channels=[0,3,5,7,8,9]
 	return prog	
