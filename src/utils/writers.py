"""
Classes for writing Borealis data to file. The different classes contain
methods for writing to different file formats.
"""

import dmap
import h5py

from utils.file_formats import SliceData


class DMAPWriter:
    """
    Class for writing to DMAP files.

    Only rawacf files are supported.
    """

    @staticmethod
    def write_record(filename: str, slice_data: SliceData, dt_str: str, file_type: str):
        """
        Write out data to a DMAP file. If the file already exists it will be overwritten.

        :param  filename:   Name of the file to write to
        :type   filename:   str
        :param  slice_data: Data to write out to the file.
        :type   slice_data: SliceData
        :param  dt_str:     A datetime timestamp of the first transmission time in the record
        :type   dt_str:     str
        :param  file_type:  Type of file to write.
        :type   file_type:  str
        """

        if file_type != "rawacf":
            raise NotImplementedError

        dmap_record = slice_data.to_dmap(dt_str)

        # todo: ensure that dmap.write_rawacf() will append to file
        dmap.write_rawacf(
            dmap_record,
            filename,
        )


class HDF5Writer:
    """
    Class for writing to HDF5 files.
    """

    @staticmethod
    def write_record(filename: str, slice_data: SliceData, dt_str: str, file_type: str):
        """
        Write out data to an HDF5 file.

        :param  filename:   Name of the file to write to
        :type   filename:   str
        :param  slice_data: Data to write out to the HDF5 file.
        :type   slice_data: SliceData
        :param  dt_str:     A datetime timestamp of the first transmission time in the record
        :type   dt_str:     str
        :param  file_type:  Type of file to write.
        :type   file_type:  str
        """
        with h5py.File(filename, "a") as f:
            group = f.create_group(dt_str)
            metadata = f.get("metadata", f.create_group("metadata"))
            slice_data.to_hdf5(group, metadata, file_type)
