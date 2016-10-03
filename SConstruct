# Copyright 2015 The Ostrich / by Itamar O
# pylint: disable=undefined-variable

"""Flavor-based project main SConstruct script with SCons shortcuts,
and support for compiling Protocol Buffers.
"""

OSTRICH_SCONS_HELP = """usage: scons [OPTION] [TARGET or FLAVOR_NAME] ...

SCons Options:
  -c, --clean, --remove       Remove specified targets and dependencies.
  -h, --help                  Print this one message and exit.
  -H, --help-options          Print SCons standard help message.
  -j N, --jobs=N              Allow N jobs at once (I recommend 8).
  -n, --no-exec, --just-print, --dry-run, --recon
                              Don't build; just print commands.
  -s, --silent, --quiet       Don't print commands.
  -u, --up, --search-up       Search up directory tree for SConstruct,
                                build targets at or below current directory.
"""

if GetOption('help'):
    # Skip it all if user just wants help
    Help(OSTRICH_SCONS_HELP)
else:
    # Get the base construction environment
    _BASE_ENV = get_base_env(tools=['default', 'protoc'])
    # Build every selected flavor
    for flavor in _BASE_ENV.flavors:
        sprint('+ Processing flavor %s ...', flavor)
        flav_bldr = FlavorBuilder(_BASE_ENV, flavor)
        flav_bldr.build()
