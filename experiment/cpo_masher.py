#!/usr/bin/python

# Mashes the cp_objects together to run as desired.
# Returns any errors in the setup.

import sys
import currentctrlprog
import zmq
import json

def setup_cp_socket():
        context=zmq.Context()
        cpsocket=context.socket(zmq.PAIR)
        cpsocket.connect("tcp://10.65.0.25:33044")
        return cpsocket

def get_cp(socket):
        update=json.dumps("UPDATE")
        socket.send(update)
        ack=socket.recv()
        reply=json.loads(ack)
        if reply=="YES":
                socket.send(json.dumps("READY"))
                new_cp=socket.recv()
                controlprog=json.loads(newcp)
                return controlprog
        else:
                return None

def create_cp(controlprog):
        """Create a control program given a controlprog class with a cpo_list and cpo_num. Check that the 
        control program can be run without errors."""
        # Get all objects and check for errors.
        error_count=0
        if (controlprog.cpo_num>=1):
                for i in range(0, cpo_num):
                        if (controlprog.cpo_list[i].selfcheck==1):
                                pass
                        else:
                                error_count=error_count+1
                # test for errors in the RCP component(s)
                if (controlprog.cpo_num>1):
                        pass
        else: # there is a problem if no RCP components.
                exit()

        return error_count


def get_seq_combos(seq_combos, cpo_num):
	seq_combos=sorted(seq_combos)
	#print seq_combos
	#if [2,4] and [1,4], then also must be [1,2] in the seq_combos list!
	i=0
	while (i<len(seq_combos)):
	#for i in range(0,(len(seq_combos)-1)):	
		#print "i: ", i,	seq_combos[i]
		k=0
		#for k in range(len(seq_combos[i])):
		while (k<len(seq_combos[i])):
			j=i+1
			while (j<len(seq_combos)):
			#for j in range(i+1,len(seq_combos)):
				#print "j: ", j, seq_combos[j]
				if seq_combos[i][k]==seq_combos[j][0]:
					add_n=seq_combos[j][1]
					#print "Adding ", add_n, " to sequence combo ", seq_combos[i]
					seq_combos[i].append(add_n) #combine the indices if there are 3+ cp_objects combining in same seq.
						#if seq_combos[i][1]>seq_combos[j][1]:
						#	seq_combos.remove([seq_combos[j][1],seq_combos[i][1]])
						#else:
					#	print seq_combos[i][1], seq_combos[j][1]
					for m in range(0,len(seq_combos[i])-1): # try all values in seq_combos[i] except the last value, which is = to add_n.
						try:
							#print "Removing sequence combo ", seq_combos[i][m], add_n
							seq_combos.remove([seq_combos[i][m],add_n]) # seq_combos[j][1] is the known last value in seq_combos[i]
						except ValueError:
						# if there is an error because that index does not exist!
							errmsg='Interfacing not Valid: CPO %d and CPO %d are a sequence combination and do not interface the same with CPO %d' % (seq_combos[i][m], seq_combos[i][k], add_n)
							sys.exit(errmsg)
					#print seq_combos
					j=j-1 # this means that the former seq_combos[j] has been deleted and there are new values at index j, so decrement before incrementing in for.
				j=j+1 # increment
			k=k+1
		i=i+1

	# now seq_combos is a list of lists, where a cp object occurs in only once in the nested list.
	for cpo in range(cpo_num):
		found=False
		for i in range(len(seq_combos)):
			for j in range(len(seq_combos[i])):
				if cpo==seq_combos[i][j]:
					found=True
					break
			if found==False:
				continue
			break
		else: # no break
			seq_combos.append([cpo]) # append the cpo on its own, is not sequence combined.
	
	return seq_combos		




def main():
	cpsocket=setup_cp_socket()
        controlprog=get_cp(cpsocket)
	# check for obvious errors by using the selfcheck function from the cp_object class.
	controlprog=currentctrlprog.experiment()
	if (controlprog.cpo_num<1):
		errmsg='Error: No objects in control program'
		sys.exit(errmsg)
	for cpo in range(controlprog.cpo_num):
		selferrs=controlprog.cpo_list[cpo].selfcheck()
		if (not selferrs): # if returned error dictionary is empty
			continue
		errmsg='Self Check Errors Occurred with Object Number : {} \nSelf Check Errors are : {}'.format(cpo, selferrs)
		sys.exit(errmsg)
	else: # no break
		print 'No Self Check Errors. Continuing...'	
	
	# check the interfacing, and combine sequences first.
	seq_combos=[]
	# check that the keys in the interface are not NONE.
	for key in controlprog.interface.keys():
		if controlprog.interface[key]=="NONE":
			errmsg='Interface keys are still default, must set key {}'.format(key)
			sys.exit(errmsg)

	for num1,num2 in controlprog.interface.keys(): # all keys in dictionary
		if (num1>=controlprog.cpo_num) or (num2>=controlprog.cpo_num) or (num1<0) or (num2<0):
			errmsg='Interfacing key (%d, %d) is not necessary and not valid' % (num1, num2)
			sys.exit(errmsg)
		if controlprog.interface[num1, num2] not in controlprog.if_type:
			errmsg='Interfacing Not Valid Type between CPO %d and CPO %d' % (num1, num2)
			sys.exit(errmsg)
		elif controlprog.interface[num1, num2]=="SAME_SEQ" or controlprog.interface[num1, num2]=="MULTI_SEQ":
			seq_combos.append([num1,num2]) # save the keys that are sequence combinations.

	seq_combos=get_seq_combos(seq_combos,controlprog.cpo_num)
		
	print "Sequence Combinations: ", seq_combos	

	# need to combine sequences.

		
	
	#		pass # some TESTS
	#	elif interface[num1, num2]=="INTTIME":
	#		pass # some TESTS
	#	elif interface[num1, num2]=="INTEGRATION":
	#		pass # some TESTS
	#	elif interface[num1, num2]=="SAME_SEQ":
	#		pass # some TESTS
	#	elif interface[num1, num2]=="MULTI_SEQ":
	#		pass # some TESTS
		
	
			# these CPO's to be treated as one, should have the following the same:
	#		for chan1 in controlprog.cpo_list[num1].channels:
	#			for chan2 in controlprog.cpo_list[num2].channels:
	#				if chan1==chan2:
	#					print "Similar channels"
	#					break
	#			else: # chan1 does not have a matching chan2,
	#				print chan1, " belongs to CPO " num1 " only"


main()
