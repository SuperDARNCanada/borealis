.. _scons:

=====
SCons
=====

``SCons`` is a utility for compiling C++ code (and other languages). It is a replacement for ``Make``, ``CMake``, and the
other myriad of tools that solve the same problem. Unlike these other tools, ``SCons`` is written in Python, and has
some nice conveniences. We use an extension of the ``SCons`` package documented at https://www.ostricher.com/tag/scons/
which adds support for some niche file types as well as build "flavours". The decision to use this extension was made
back when Borealis contained ``.proto`` and ``.cu`` files that required ``protobuf`` and ``nvcc`` for compilation. These files
have since been removed, but the build "flavours" have been quite useful so we continue to use the extension.

``SCons`` requires a few files for specifying how to compile code files. First, a file called ``SConstruct`` is included
at the top level of the Borealis repository. This is the ``SCons`` equivalent of a ``Makefile``.

In each folder with code files that need compiling, an additional ``SConscript`` file must be present.
This file specifies the directives for how to compile the code files; i.e., compile into a program or a library,
what files to include, and what libraries must be linked for compilation. ``SCons`` searches for ``SConscript`` files
when the ``scons`` command is invoked.

--------------
Build Flavours
--------------

Borealis has two build flavours: ``release``, and ``debug``.

``release`` is the optimized build intended for operation of the radar. This flavour limits the logging
messages in the C++ code to a readable level. When compiled in ``debug`` mode, time profiling is conducted and
logged as well, for critical sections of the code.

---------
Compiling
---------

Compilation is easy: ``scons [release|debug]``, while in the directory that contains the ``SConstruct`` file.

To remove the compiled files, run ``scons -c`` and the build directory will be cleared. Note that the build
directory is split by flavour, so you can build in both ``release`` and ``debug`` modes sequentially and have both sets of
output files available.

Within the ``build/`` directory, there will be a directory for each build flavour. Within these directories, the
code files are first copied in, then built in-place, so the directory structure matches the Borealis repository
(including only the code files requiring compilation).
