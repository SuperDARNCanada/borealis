Borealis - A control system for USRP based digital radars
=========================================================

## Low Bandwidth Modification
The default bandwidth for Borealis is 5 MHz, this modification shrinks the bandwidth to 500 kHz. This fork also fixes an issue where the USRP Driver will send timed USRP packets that occur in the past by bumping up the command delay time.

Run the code using:
```
$ source mode release
$ scons
$ python3 steamed_hams.py normalscan_low_bw release common
$ screen -r
```

To kill the application use:
```
$ screen -list
$ screen -X -S <ID> quit
```

See our latest documentation here: https://borealis.readthedocs.io/en/latest/ 

Software setup specifically can be found here: https://borealis.readthedocs.io/en/latest/setup_software.html
