# !/usr/bin/python3
# Copyright 2015 The Ostrich / by Itamar O
# Copyright 2016 SuperDARN

# The MIT License (MIT)

# Copyright (c) 2014 The Ostrich
# Modified 2016 SuperDARN

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""SCons site config script"""

import os
from site_utils import module_dirs_generator

# Directory for build process outputs (object files etc.)
_BUILD_BASE = "build"
# Directory where binary programs are installed in (under $build_base/$flavor)
_BIN_SUBDIR = "bin"

# List of cached modules to save processing for second call and beyond
_CACHED_MODULES = list()


def modules():
    """Generate modules to build.

    Each module is a directory with a SConscript file.
    """
    if not _CACHED_MODULES:
        # Build the cache
        def build_dir_skipper(dirpath):
            """Return True if `dirpath` is the build base dir."""
            return os.path.normpath(_BUILD_BASE) == os.path.normpath(dirpath)

        def hidden_dir_skipper(dirpath):
            """Return True if `dirpath` last dir component begins with '.'"""
            last_dir = os.path.basename(dirpath)
            return last_dir.startswith(".")

        for module_path in module_dirs_generator(
            max_depth=7,
            followlinks=False,
            dir_skip_list=[build_dir_skipper, hidden_dir_skipper],
            file_skip_list=".noscons",
        ):
            _CACHED_MODULES.append(module_path)
    # Yield modules from cache
    for module in _CACHED_MODULES:
        yield module


# Dictionary of flavor-specific settings that should override values
#  from the base environment (using env.Replace).
# `_common` is reserved for settings that apply to the base env.
ENV_OVERRIDES = {
    "_common": dict(
        # Use clang compiler by default
        CC="gcc",
        CXX="g++",
        # Path for installed binary programs
        BINDIR=os.path.join("$BUILDROOT", _BIN_SUBDIR),
    ),
    "debug": dict(
        BUILDROOT=os.path.join(_BUILD_BASE, "debug"),
    ),
    "release": dict(
        BUILDROOT=os.path.join(_BUILD_BASE, "release"),
    ),
}

# Dictionary of flavor-specific settings that should extend values
#  from the base environment (using env.Append).
# `_common` is reserved for settings that apply to the base env.

ENV_EXTENSIONS = {
    "_common": dict(
        # Common flags for all C++ builds
        CCFLAGS=["-Wall"],
        CXXFLAGS=["-std=c++11"],
        CFLAGS=["-std=c99"],
        # Modules should be able to include relative to build root dir
        CPPPATH=["#$BUILDROOT"],
    ),
    "debug": dict(
        # Extra flags for debug C++ builds
        CCFLAGS=["-g", "-DDEBUG"],
    ),
    "release": dict(
        # Extra flags for release C++ builds
        CCFLAGS=["-DNDEBUG", "-O3"],
    ),
}


def flavors():
    """Generate supported flavors.

    Each flavor is a string representing a flavor entry in the
    override / extension dictionaries above.
    Each flavor entry must define atleast "BUILDROOT" variable that
    tells the system what's the build base directory for that flavor.
    """
    # Use the keys from the env override / extension dictionaries
    for flavor in set(list(ENV_EXTENSIONS.keys()) + list(ENV_OVERRIDES.keys())):
        # Skip "hidden" records
        if not flavor.startswith("_"):
            yield flavor


def main():
    """Main procedure - print out a requested variable (value per line)"""
    import sys

    if 2 == len(sys.argv):
        var = sys.argv[1].lower()
        items = list()
        if var in ("flavors",):
            items = flavors()
        elif var in ("modules",):
            items = modules()
        elif var in ("build", "build_dir", "build_base"):
            items = [_BUILD_BASE]
        elif var in ("bin", "bin_subdir"):
            items = [_BIN_SUBDIR]
        # print out the item values
        for val in items:
            print(val)


if "__main__" == __name__:
    main()
