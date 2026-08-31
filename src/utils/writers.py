"""
Classes for writing Borealis data to disk in different file formats.
"""

import datetime as dt
import dmap
import h5py

from .file_formats import SliceData


class DMAPWriter:
    """
    Class for writing to DMAP files.

    Only rawacf files are supported.
    """

    @staticmethod
    def write_record(filename: str, slice_data: SliceData, file_type: str):
        """
        Write out data to a DMAP file.

        :param  filename:   Name of the file to write to
        :type   filename:   str
        :param  slice_data: Data to write out to the file
        :type   slice_data: SliceData
        :param  file_type:  Type of file to write
        :type   file_type:  str
        """

        if file_type != "rawacf":
            raise NotImplementedError

        dmap_record = slice_data.to_dmap()
        dmap.write_rawacf(dmap_record, filename)


class HDF5Writer:
    """
    Class for writing to HDF5 files.
    """

    @staticmethod
    def write_record(filename: str, slice_data: SliceData, file_type: str):
        """
        Write out data to an HDF5 file.

        :param  filename:   Name of the file to write to
        :type   filename:   str
        :param  slice_data: Data to write out to the HDF5 file
        :type   slice_data: SliceData
        :param  file_type:  Type of file to write
        :type   file_type:  str
        """
        with h5py.File(filename, "a") as f:
            timestamp = dt.datetime.fromtimestamp(
                slice_data.sqn_timestamps[0], dt.timezone.utc
            )
            dt_str = timestamp.strftime("%Y%m%d-%H%M-%S.%f")
            group = f.create_group(dt_str)
            if "metadata" in f.keys():
                metadata = f.get("metadata")
            else:
                metadata = f.create_group("metadata")
            slice_data.to_hdf5(group, metadata, file_type)
