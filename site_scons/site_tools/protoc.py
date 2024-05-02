# Copyright (c) 2009  Scott Stafford
# Copyright 2014 The Ostrich / by Itamar O
# Copyright 2016 SuperDARN
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY
# KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

"""
protoc.py: Protoc Builder for SCons

This Builder invokes protoc to generate C++ and Python from a .proto file.
NOTE: Java is not currently supported.

Derived from original work by Scott Stafford
(http://www.scons.org/wiki/ProtocBuilder)
"""

__author__ = "Itamar Ostricher"

import os
import re

import SCons

_PROTOCS = "protoc"
_PROTOSUFFIX = ".proto"

_PROTOC_SCANNER_RE = re.compile(r"^import\s+\"(.+\.proto)\"\;$", re.M)


def protoc_emitter(target, source, env):
    """Return list of targets generated by Protoc builder for source."""
    for src in source:
        proto = os.path.splitext(str(src))[0]
        if env["PROTOCPPOUT"]:
            target.append("%s.pb.cc" % (proto))
            target.append("%s.pb.h" % (proto))
        if env["PROTOPYOUT"]:
            target.append("%s_pb2.py" % (proto))
    return target, source


def protoc_scanner(node, env, _):
    """Return list of file nodes that `node` imports"""
    contents = node.get_text_contents()
    # If build location different from sources location,
    #  get the destination base dir as the base for imports.
    nodepath = str(node.path)
    srcnodepath = str(node.srcnode())
    src_pos = nodepath.find(srcnodepath)
    base_path = src_pos and nodepath[: src_pos - 1] or ""
    imports = [
        os.path.join(base_path, imp) for imp in _PROTOC_SCANNER_RE.findall(contents)
    ]
    return env.File(imports)


def generate(env):
    """Add Builders, Scanners and construction variables
    for protoc to the build Environment."""
    try:
        bldr = env["BUILDERS"]["Protoc"]
    except KeyError:
        action = SCons.Action.Action("$PROTOCOM", "$PROTOCOMSTR")
        bldr = SCons.Builder.Builder(
            action=action, emitter=protoc_emitter, src_suffix="$PROTOCSRCSUFFIX"
        )
        env["BUILDERS"]["Protoc"] = bldr

    # pylint: disable=bad-whitespace
    env["PROTOC"] = env.Detect(_PROTOCS) or "protoc"
    env["PROTOCFLAGS"] = SCons.Util.CLVar("")
    env["PROTOCSRCSUFFIX"] = _PROTOSUFFIX
    # Default proto search path is same dir
    env["PROTOPATH"] = ["."]
    # Default CPP output in same dir
    env["PROTOCPPOUT"] = "."
    # No default Python output
    env["PROTOPYOUT"] = ""
    proto_cmd = ["$PROTOC"]
    proto_cmd.append('${["--proto_path=%s"%(x) for x in PROTOPATH]}')
    proto_cmd.append("$PROTOCFLAGS")
    proto_cmd.append('${PROTOCPPOUT and "--cpp_out=%s"%(PROTOCPPOUT) or ""}')
    proto_cmd.append('${PROTOPYOUT and "--python_out=%s"%(PROTOPYOUT) or ""}')
    proto_cmd.append("${SOURCES}")
    env["PROTOCOM"] = " ".join(proto_cmd)

    # Add the proto scanner (if it wasn't added already)
    env.AppendUnique(
        SCANNERS=SCons.Scanner.Scanner(function=protoc_scanner, skeys=[_PROTOSUFFIX])
    )


def exists(env):
    """Return True if `protoc` tool exists in the system."""
    return env.Detect(_PROTOCS)
