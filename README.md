Borealis - A control system for USRP based digital radars formerly known as placeholderOS
=========================================================================================

## Dependencies
So far we need to have the following things installed (tested on OpenSuSe 42.2). See the 'install_radar_deps.sh' bash scripts for more information. 

* UHD library https://github.com/EttusResearch/uhd and http://files.ettus.com/manual/page_build_guide.html for build instructionsi
* CUDA Toolkit https://developer.nvidia.com/cuda-toolkit http://developer.download.nvidia.com/compute/cuda/repos/opensuse422/x86_64/cuda-repo-opensuse422-9.0.176-1.x86_64.rpm
* Protobuf library + compiler https://github.com/google/protobuf (Note that at the time of writing, protobuf 2 is latest stable release, but you need protobuf 3)
* 'pip install protobuf' for python protobuf bindings
* scons ('zypper in scons' on openSuSe)
* zmq ('zypper in zeromq-devel' and 'zypper in cppzmq-devel' on openSuSe)

## Environment
In order to set up the correct environment, we use a script called 'mode' in the parent directory, in conjunction with scons.
To set up debug environment:

source mode debug

And for release:

source mode release

This will add the proper directory to your environment PATH variable for the built binaries. 
It also adds specific flags for the C++ compiler for either debug or release. See the python
script at site_scons/site_config.py for all details.

In order to see which modules the site config has, you can query the site_config.py script like so:
	./site_config.py modules

Or to see the bin subdirectory: 
	./site_config.py bin

To see the build subdirectory:
	./site_config.py build

To see the various build flavours available (typically debug and release):
	./site_config.py flavors

** NOTE ** You need to execute the script exactly as above (no 'python' in front), or the output won't show up.
This means the script must be executable.

## Installation

Now to build borealis (after 'source mode release/debug'):
scons -n  (for dry run)
scons


