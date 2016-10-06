#!/usr/bin/python

# Mashes the cp_objects together to run as desired.
# Returns any errors in the setup.

from currentcontrolprog import controlprog


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


def main():
	# check for obvious errors.
	for cpo in range(controlprog.cpo_num):
		if !(controlprog.cpo_list[cpo].selfcheck): # if returned error dictionary is empty
			continue
		print 'Self Check Errors Occurred with Object Number : ', cpo
	 	quit()
	else:
		print 'No Self Check Errors. Continuing...'	
	
	# check the interfacing.
	seq_combo=[]
	for num1,num2 in controlprog.interface: # all keys in dictionary
		if controlprog.interface[num1, num2] not in controlprog.if_type:
			print 'Interfacing Not Valid Type between CPO ', num1, ' and CPO ', num2
			quit()
		elif interface[num1, num2]=="SAME_SEQ" or interface[num1, num2]="MULTI_SEQ":
			seq_combo.append([num1,num2]) # save the keys that are sequence combinations.
					
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
			for chan1 in controlprog.cpo_list[num1].channels:
				for chan2 in controlprog.cpo_list[num2].channels:
					if chan1==chan2:
						print "Similar channels"
						break
				else: # chan1 does not have a matching chan2,
					print chan1, " belongs to CPO " num1 " only"

