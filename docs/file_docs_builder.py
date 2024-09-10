"""
Generates a .rst file for each type of file that Borealis can produce, detailing the fields and metadata
of each field.
"""

import copy
from dataclasses import fields
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]) + "/src/utils/")
from file_formats import SliceData


field_format = ":{name}: {description}\n\n" "{metadata}"
metadata_format = "   * {key}: ``{value}``\n"


def generate_for_file(file_type: str):
    preamble = copy.deepcopy(SliceData.__doc__).split("\n")[3:]
    preamble = "\n".join(preamble)
    docstring = (
        f".. _{file_type}:\n\n"
        f"{'-'*len(file_type)}\n"
        f"{file_type.lower()}\n"
        f"{'-'*len(file_type)}\n"
        f"{preamble}\n\n"
        f"Fields\n"
        f"------\n"
    )

    for f in fields(SliceData):
        if file_type not in f.metadata.get("groups"):
            continue

        description = f.metadata["description"]
        metadata = ""
        for k, v in sorted(f.metadata.items(), key=lambda item: item[0]):
            if k in ["groups", "description"]:
                continue
            metadata += metadata_format.format(key=k, value=v)
        field_str = field_format.format(
            name=f.name, description=description, metadata=metadata
        )

        docstring += f"{field_str}\n\n"
    return docstring


def write_rst_file(outdir: str, file_type: str):
    outfile = f"{outdir}/{file_type}.rst"

    with open(outfile, "w") as ofile:
        ofile.write(generate_for_file(file_type))


def main():
    file_types = ["antennas_iq", "bfiq", "rawacf", "rawrf"]
    outdir = "source/"
    for ftype in file_types:
        write_rst_file(outdir, ftype)


if __name__ == "__main__":
    main()
