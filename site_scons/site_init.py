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

"""SCons site init script - automatically imported by SConstruct"""

from collections import defaultdict
import os

from SCons import Node
from SCons.Errors import StopError

from site_config import flavors, modules, ENV_OVERRIDES, ENV_EXTENSIONS
from site_utils import listify, path_to_key, nop, sprint


def get_base_env(*args, **kwargs):
    """Initialize and return a base construction environment.

    All args received are passed transparently to SCons Environment init.
    """
    # Initialize new construction environment
    env = Environment(*args, **kwargs)  # noqa: F821
    # If a flavor is activated in the external environment - use it
    if "BUILD_FLAVOR" in os.environ:
        active_flavor = os.environ["BUILD_FLAVOR"]
        if active_flavor not in flavors():
            raise StopError("%s (from env) is not a known flavor." % (active_flavor))
        sprint('Using active flavor "%s" from your environment', active_flavor)
        env.flavors = [active_flavor]
    else:
        # If specific flavor target specified, skip processing other flavors
        # Otherwise, include all known flavors
        env.flavors = (
            set(flavors()).intersection(COMMAND_LINE_TARGETS)  # noqa: F821
            or flavors()
        )
    # Perform base construction environment customizations from site_config
    if "_common" in ENV_OVERRIDES:
        env.Replace(**ENV_OVERRIDES["_common"])
    if "_common" in ENV_EXTENSIONS:
        env.Append(**ENV_EXTENSIONS["_common"])
    return env


class FlavorBuilder(object):
    """Build manager class for flavor."""

    _key_sep = "::"

    @classmethod
    def lib_key(cls, module, target_name):
        """Return unique identifier for target `target_name` in `module`"""
        return "%s%s%s" % (path_to_key(module), cls._key_sep, path_to_key(target_name))

    @classmethod
    def is_lib_key(cls, str_to_check):
        """Return True if `str_to_check` is a library identifier string"""
        return cls._key_sep in str_to_check

    def __init__(self, base_env, flavor):
        """Initialize a build manager instance for flavor.

        @param base_env     Basic construction environment to start from
        @param flavor       The flavor to process
        """
        self._flavor = flavor
        # Create construction env clone for flavor customizations
        self._env = base_env.Clone()
        # Initialize shared libraries dictionary
        self._libs = dict()
        # Initialize programs dictionary
        self._progs = defaultdict(list)
        # Apply flavored env overrides and customizations
        if flavor in ENV_OVERRIDES:
            self._env.Replace(**ENV_OVERRIDES[flavor])
        if flavor in ENV_EXTENSIONS:
            self._env.Append(**ENV_EXTENSIONS[flavor])
        # Support using the flavor name as target name for its related targets
        self._env.Alias(flavor, "$BUILDROOT")

    def build(self):
        """Build flavor using two-pass strategy."""
        # First pass over all modules - process and collect library targets
        for module in modules():
            shortcuts = dict(
                Lib=self._lib_wrapper(self._env.Library, module),
                StaticLib=self._lib_wrapper(self._env.StaticLibrary, module),
                SharedLib=self._lib_wrapper(self._env.SharedLibrary, module),
                Prog=nop,
            )
            self._env.SConscript(
                os.path.join(module, "SConscript"),
                variant_dir=os.path.join("$BUILDROOT", module),
                exports=shortcuts,
            )

        # Second pass over all modules - process program targets
        shortcuts = dict()
        for nop_shortcut in ("Lib", "StaticLib", "SharedLib"):
            shortcuts[nop_shortcut] = nop

        for module in modules():
            sprint("|- Second pass: Reading module %s ...", module)
            shortcuts["Prog"] = self._prog_wrapper(module)
            self._env.SConscript(
                os.path.join(module, "SConscript"),
                variant_dir=os.path.join("$BUILDROOT", module),
                exports=shortcuts,
            )

        # Add install targets for programs from all modules
        for module, prog_nodes in self._progs.items():
            for prog in prog_nodes:
                assert isinstance(prog, Node.FS.File)
                # If module is hierarchical, replace pathseps with periods
                bin_name = path_to_key("%s" % (prog.name))
                self._env.InstallAs(os.path.join("$BINDIR", bin_name), prog)
        # Support using the flavor name as target name for its related targets
        self._env.Alias(self._flavor, "$BUILDROOT")

    def _lib_wrapper(self, bldr_func, module):
        """Return a wrapped customized flavored library builder for module.

        @param  builder_func        Underlying SCons builder function
        @param  module              Module name
        """

        def build_lib(lib_name, sources=None, **kwargs):
            """Customized library builder.

            @param  lib_name    Library name
            @param  sources     Source file (or list of source files)
            kwargs params:
            """
            # Create unique library key from module and library name
            lib_key = self.lib_key(module, lib_name)
            assert lib_key not in self._libs
            sources = listify(sources)
            # Store resulting library node in shared dictionary
            self._libs[lib_key] = bldr_func(lib_name, sources, **kwargs)

        return build_lib

    def _prog_wrapper(self, module, default_install=True):
        """Return a wrapped customized flavored program builder for module.

        @param  module              Module name
        @param  default_install     Whether built program nodes should be
                                    installed in bin-dir by default
        """

        def build_prog(prog_name, sources=None, with_libs=None, **kwargs):
            """Customized program builder.

            @param  prog_name   Program name
            @param  sources     Source file (or list of source files)
            @param  with_libs   Library name (or list of library names) to
                                link with.
            kwargs params:
            @param  install     Binary flag to override default value from
                                closure (`default_install`).
            """
            sources = listify(sources)
            install_flag = kwargs.pop("install", default_install)
            # Process library dependencies - add libs specified in `with_libs`
            for lib_name in listify(with_libs):
                lib_keys = listify(self._get_matching_lib_keys(lib_name))
                if len(lib_keys) == 1:
                    # Matched internal library
                    lib_key = lib_keys[0]
                    # Extend prog sources with library nodes
                    sources.extend(self._libs[lib_key])
                elif len(lib_keys) > 1:
                    # Matched multiple internal libraries - probably bad!
                    raise StopError(
                        'Library identifier "%s" matched %d '
                        "libraries (%s). Please use a fully "
                        "qualified identifier instead!"
                        % (lib_name, len(lib_keys), ", ".join(lib_keys))
                    )
                else:  # empty lib_keys
                    raise StopError(
                        'Library identifier "%s" didn\'t match '
                        "any library. Is it a typo?" % (lib_name)
                    )
            # Build the program and add to prog nodes dict if installable
            prog_nodes = self._env.Program(prog_name, sources, **kwargs)
            if install_flag:
                # storing each installable node in a dictionary instead of
                #  defining InstallAs target on the spot, because there's
                #  an "active" variant dir directive messing with paths.
                self._progs[module].extend(prog_nodes)

        return build_prog

    def _get_matching_lib_keys(self, lib_query):
        """Return list of library keys for given library name query.

        A "library query" is either a fully-qualified "Module::LibName" string
         or just a "LibName".
        If just "LibName" form, return all matches from all modules.
        """
        if self.is_lib_key(lib_query):
            # It's a fully-qualified "Module::LibName" query
            if lib_query in self._libs:
                # Got it. We're done.
                return [lib_query]
        else:
            # It's a target-name-only query. Search for matching lib keys.
            lib_key_suffix = "%s%s" % (self._key_sep, lib_query)
            return [
                lib_key for lib_key in self._libs if lib_key.endswith(lib_key_suffix)
            ]
