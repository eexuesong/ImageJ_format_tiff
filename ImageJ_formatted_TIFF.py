"""
Reads and writes tiff stack following ImageJ format pseudo-Tiff.

V2.0 Xuesong Li 04/13/2023:
"""

import numpy
import sys
import os
import math
import json
import time
import tifffile
import warnings
from pprint import pprint

numpy_version = numpy.version.version.split('.')


def WriteTifStack(numpy_array, filename, resolution=0.05, spacing=0.1, imformat='z', endian='little',
                  coerce_64bit_to_32bit=True):
    assert len(numpy_array.shape) >= 2
    dtype = numpy_array.dtype

    if int(numpy_version[1]) >= 20:
        bool_type = bool
    else:
        bool_type = numpy.bool

    if coerce_64bit_to_32bit and numpy_array.dtype in (numpy.float64, numpy.int64, numpy.uint64):
        if numpy_array.dtype == numpy.float64:
            dtype = numpy.dtype('float32')
        elif numpy_array.dtype == numpy.int64:
            dtype = numpy.dtype('int32')
        elif numpy.dtype == numpy.uint64:
            dtype = numpy.dtype('uint32')
    elif numpy_array.dtype == bool_type:  # Coerce boolean arrays to uint8
        dtype = numpy.dtype('uint8')
    else:
        dtype = numpy_array.dtype

    dimension = numpy_array.shape
    width = dimension[-1]
    height = dimension[-2]
    dimension_tzc = dimension[0:-2]

    channels = 1
    slices = 1
    frames = 1
    if len(dimension_tzc) == 0:
        # 2D slice
        pass
    else:
        # 3D, 4D or 5D stack
        assert len(dimension_tzc) == len(imformat)
        if len(dimension_tzc) == 1:
            if imformat == ('c' or 'C'):
                channels = dimension_tzc[0]
                print(imformat)
            elif imformat == ('z' or 'Z'):
                slices = dimension_tzc[0]
            elif imformat == ('t' or 'T'):
                frames = dimension_tzc[0]
            else:
                raise UserWarning("'imformat' must be 'c','z','t','zc','tc','tz' or 'tzc'.")
        elif len(dimension_tzc) == 2:
            if imformat == ('zc' or 'ZC'):
                (slices, channels) = dimension_tzc
            elif imformat == ('tc' or 'TC'):
                (frames, channels) = dimension_tzc
            elif imformat == ('tz' or 'TZ'):
                (frames, slices) = dimension_tzc
            else:
                raise UserWarning("'imformat' must be 'c','z','t','zc','tc','tz' or 'tzc'.")
        elif len(dimension_tzc) == 3:
            if imformat == ('tzc' or 'TZC'):
                (frames, slices, channels) = dimension_tzc
            else:
                raise UserWarning("'imformat' must be 'c','z','t','zc','tc','tz' or 'tzc'.")
        else:
            raise UserWarning("'imformat' has incorrect character number.")

    minval = numpy_array.min()
    maxval = numpy_array.max()
    header = write_IFD(filename=filename, endian=endian, width=width, height=height, data_type=dtype, channels=channels,
                       slices=slices, frames=frames, spacing=spacing, min=minval, max=maxval, resolution=resolution)

    with open(file=filename, mode='ab') as f:
        # "a": Append - Opens a file for writing, appending to the end of file if it exists
        # "b": Binary - Open in binary mode. (e.g. images)
        if dtype != numpy_array.dtype:  # We have to coerce to a different dtype
            if sys.byteorder != endian:
                numpy_array.astype(dtype=dtype, copy=True).byteswap(inplace=False).tofile(f)
            else:
                numpy_array.astype(dtype=dtype, copy=True).tofile(f)
        else:
            if sys.byteorder != endian:
                numpy_array.byteswap(inplace=False).tofile(f)
            else:
                numpy_array.tofile(f)

# Obsolete
# def ReadTifStack(filename):
#     """
#     Load a tif into memory and return it as a numpy array.
#     """
#     # First determine whether it is a ImageJ formatted TIFF file (New version: Xuesong 04/19/2023)
#     image_info = tifffile.TiffFile(filename)
#     # number of pages in the file
#     if len(image_info.pages) > 1:
#         # print("Not a ImageJ formatted Tiff file.")
#         ImageJ_formatted_TIFF_flag = False
#     else:
#         ImageJ_formatted_TIFF_flag = True
#
#     # Read and parse tiff header first
#     with open(file=filename, mode='rb') as f:
#         # "r": Read - Default value. Opens a file for reading, error if the file does not exist
#         # "b": Binary - Open in binary mode. (e.g. images)
#         header = parse_tif(File_object=f, verbose=False)
#         ImageWidth = header.ImageWidth
#         ImageLength = header.ImageLength
#         BitsPerSample = header.BitsPerSample
#         Depth_estimated = math.floor(os.path.getsize(filename) / (ImageWidth * ImageLength * BitsPerSample / 8))
#
#         # Calculate slices from header.ImageDescription (Old version)
#         # k1 = header.ImageDescription.find('images=')
#         # ImageDescription_crop = header.ImageDescription[k1:]
#         # k2 = ImageDescription_crop.find('\n')
#         # if (k1 != -1) and (k2 != -1):
#         #     Depth = int(header.ImageDescription[k1 + 7: k1 + k2])
#         # else:
#         #     # print("Did not find 'images=' in ImageDescription. Try to get it from filesize.")
#         #     Depth = int(os.path.getsize(filename) / (ImageWidth * ImageLength * (header.BitsPerSample / 8)))
#
#         """
#         Determine the numpy data_type from the TIF header.BitsPerSample and header.SampleFormat
#         Allocate our numpy array, and load data into our array from disk, one image at a time.
#         """
#         dtype = {
#                     1: 'uint',
#                     2: 'int',
#                     3: 'float',
#                     4: 'undefined',
#                 }[header.SampleFormat] + ascii(header.BitsPerSample)
#         try:
#             dtype = getattr(numpy, dtype)
#         except AttributeError:
#             raise UserWarning("ReadTifStack does not support data format: {0}".format(dtype))
#
#         if ImageJ_formatted_TIFF_flag:
#             # Initialize reading buffer and parameters
#             if header.images is None:
#                 Depth = Depth_estimated
#             else:
#                 Depth = header.images
#                 # Double check
#                 if Depth != Depth_estimated:
#                     raise UserWarning(
#                         "Image number calculated from filesize is inconsistent with 'images=' in ImageDescription.")
#
#             PixelNum = ImageWidth * ImageLength
#             ByteCounts = int(PixelNum * BitsPerSample / 8)
#             f.seek(header.StripOffsets, 0)
#             # Old version
#             # Stack = numpy.zeros(Depth * PixelNum, dtype=dtype)
#             # for t in range(Depth):
#             #     Stack[t * PixelNum: (t + 1) * PixelNum] = numpy.fromfile(file=f, dtype=dtype, count=PixelNum, sep='', offset=0)
#
#             # New version (Xuesong Li: 04/20/2023)
#             Stack = numpy.fromfile(file=f, dtype=dtype, count=Depth * ByteCounts, sep='', offset=0)
#
#             if sys.byteorder != header.endian:
#                 Stack = Stack.byteswap()
#             Stack = Stack.reshape([Depth, ImageLength, ImageWidth])
#         else:
#             Stack = numpy.zeros((len(image_info.pages), ImageLength, ImageWidth), dtype=dtype)
#             for page in range(len(image_info.pages)):
#                 Stack[page, :, :] = tifffile.imread('temp.tif', key=page)
#
#         # Further reshape Stack based on "channels", "slices" and "frames"
#         if header.channels is None:
#             channels = 1
#         else:
#             channels = header.channels
#
#         if header.slices is None:
#             slices = 1
#         else:
#             slices = header.slices
#
#         if header.frames is None:
#             frames = 1
#         else:
#             frames = header.frames
#
#         if Depth != (channels * slices * frames):
#             if (channels == 1) and (slices == 1) and (frames == 1):
#                 slices = Depth
#                 header.slices = slices
#             else:
#                 raise UserWarning("channels * slices * frames dose not match total image number.")
#
#         # Reshape into final format
#         Stack = Stack.reshape([frames, slices, channels, ImageLength, ImageWidth])
#         # Remove unnecessary dimension(s)
#         Stack = Stack.squeeze()
#     return Stack, header


def ReadTifStack(filename):
    """
    Load a tif into memory and return it as a numpy array.
    """
    # Get tiff file header
    header = get_header(filename=filename)

    """
    Determine the numpy data_type from the TIF header.BitsPerSample and header.SampleFormat
    """
    dtype = {
                1: 'uint',
                2: 'int',
                3: 'float',
                4: 'undefined',
            }[header.SampleFormat] + ascii(header.BitsPerSample)
    try:
        dtype = getattr(numpy, dtype)
    except AttributeError:
        raise UserWarning("ReadTifStack does not support data format: {0}".format(dtype))

    # Determine whether it is a ImageJ formatted TIFF file (New version)
    image_info = tifffile.TiffFile(filename)
    if len(image_info.pages) > 1:
        # print("Not a ImageJ formatted Tiff file.")
        if header.MicroManagerBigTiff:
            Stack = numpy.zeros([len(image_info.pages), header.ImageLength, header.ImageWidth], dtype=dtype)
            page_index = 0
            with tifffile.TiffFile(filename) as tif:
                for page in tif.pages:
                    Stack[page_index, :, :] = page.asarray()
                    page_index = page_index + 1
        else:
            Stack = tifffile.imread(filename)
    else:
        """
        Allocate our numpy array, and load data into our array from disk, one image at a time.
        """
        PixelNum = header.ImageWidth * header.ImageLength
        ByteCounts = int(PixelNum * header.BitsPerSample / 8)
        with open(file=filename, mode='rb') as f:
            # "r": Read - Default value. Opens a file for reading, error if the file does not exist
            # "b": Binary - Open in binary mode. (e.g. images)
            f.seek(header.StripOffsets, 0)
            Stack = numpy.fromfile(file=f, dtype=dtype, count=header.images * ByteCounts, sep='', offset=0)

        if sys.byteorder != header.endian:
            Stack = Stack.byteswap()
        Stack = Stack.reshape([header.images, header.ImageLength, header.ImageWidth])

        # Further reshape Stack based on "channels", "slices" and "frames"
        if header.channels is None:
            channels = 1
        else:
            channels = header.channels

        if header.slices is None:
            slices = 1
        else:
            slices = header.slices

        if header.frames is None:
            frames = 1
        else:
            frames = header.frames

        # Reshape into final format
        Stack = Stack.reshape([frames, slices, channels, header.ImageLength, header.ImageWidth])
        # Remove unnecessary dimension(s)
        Stack = Stack.squeeze()

    return Stack, header


def get_header(filename):
    # First read and parse tiff header
    with open(file=filename, mode='rb') as f:
        # "r": Read - Default value. Opens a file for reading, error if the file does not exist
        # "b": Binary - Open in binary mode. (e.g. images)
        header = parse_tif(File_object=f, verbose=False)
        ImageWidth = header.ImageWidth
        ImageLength = header.ImageLength
        BitsPerSample = header.BitsPerSample

    # Then determine whether it is a ImageJ formatted TIFF file
    image_info = tifffile.TiffFile(filename)
    images_estimated = math.floor(os.path.getsize(filename) / (ImageWidth * ImageLength * BitsPerSample / 8))

    if len(image_info.pages) > 1:
        # print("Not a ImageJ formatted Tiff file.")
        if header.images is None:
            header.images = len(image_info.pages)
        if header.images != len(image_info.pages):
            # header.images = len(image_info.pages)
            header.MicroManagerBigTiff = True
            warnings.warn("Image number get from tifffile.TiffFile is inconsistent with 'images=' in ImageDescription. It is part of a MicroManager big tiff (>4GB).")
    else:
        if header.images is None:
            header.images = images_estimated

    # Double check
    if header.images != images_estimated:
        warnings.warn("Image number calculated from filesize is inconsistent with 'images=' in ImageDescription.")

    # Check header.images == channels * slices * frames
    if header.channels is None:
        channels = 1
    else:
        channels = header.channels

    if header.slices is None:
        slices = 1
    else:
        slices = header.slices

    if header.frames is None:
        frames = 1
    else:
        frames = header.frames

    if header.images != (channels * slices * frames):
        if (channels == 1) and (slices == 1) and (frames == 1):
            header.slices = header.images
            header.slices = slices
        else:
            raise UserWarning("channels * slices * frames dose not match total image number.")

    return header


def ReadTifStack_1d(filename):
    """
    Load a tif into memory and return it as a numpy array.
    """
    # Get tiff file header
    header = get_header(filename=filename)

    # Determine whether it is a ImageJ formatted TIFF file (New version)
    image_info = tifffile.TiffFile(filename)
    if len(image_info.pages) > 1:
        # print("Not a ImageJ formatted Tiff file.")
        Stack = tifffile.imread(filename)
        Stack = Stack.flatten()
    else:
        """
        Determine the numpy data_type from the TIF header.BitsPerSample and header.SampleFormat
        """
        dtype = {
                    1: 'uint',
                    2: 'int',
                    3: 'float',
                    4: 'undefined',
                }[header.SampleFormat] + ascii(header.BitsPerSample)
        try:
            dtype = getattr(numpy, dtype)
        except AttributeError:
            raise UserWarning("ReadTifStack does not support data format: {0}".format(dtype))

        """
        Allocate our numpy array, and load data into our array from disk, one image at a time.
        """
        PixelNum = header.ImageWidth * header.ImageLength
        ByteCounts = int(PixelNum * header.BitsPerSample / 8)
        with open(file=filename, mode='rb') as f:
            # "r": Read - Default value. Opens a file for reading, error if the file does not exist
            # "b": Binary - Open in binary mode. (e.g. images)
            f.seek(header.StripOffsets, 0)
            Stack = numpy.fromfile(file=f, dtype=dtype, count=header.images * ByteCounts, sep='', offset=0)

        if sys.byteorder != header.endian:
            Stack = Stack.byteswap()

    return Stack, header


def write_IFD(filename, endian='little', width=1, height=1, data_type=numpy.dtype('uint16'), channels=1, slices=1,
              frames=1, spacing=0.1, min=None, max=None, resolution=0.05, rewrite_flag=False):
    # We'll structure our TIF in the following way (same as how ImageJ saves tiff stack):
    # 8-byte Image File Header
    # 1st image file directory (IFD)
    # Image description (~100 bytes)
    # Image XResolution (Two 32-bit unsigned integers, 8 bytes)
    # Image YResolution (Two 32-bit unsigned integers, 8 bytes)
    # 1st image data
    # 2nd image data
    # ...
    # last image data

    if rewrite_flag:
        open_mode = 'rb+'
        '''
        "r+": Read and Write - Opens a file for reading and writing, creates the file if it does not exist
        "b": Binary - Open in binary mode. (e.g. images)
        '''
    else:
        open_mode = 'wb'
        '''
        "w": Write - Opens a file for writing, creates the file if it does not exist
        "b": Binary - Open in binary mode. (e.g. images)
        '''

    with open(file=filename, mode=open_mode) as f:
        # region Write Tiff Header into file
        if endian == 'little':
            f.write(b'\x49\x49\x2A\x00')  # little-endian (Intel format) order
        else:
            f.write(b'\x4D\x4D\x00\x2A')  # big-endian (Motorola format) order

        if sys.byteorder != endian:
            swapbytes_flag = True
        else:
            swapbytes_flag = False

        IFDOffset = numpy.zeros(4, dtype=numpy.uint8)
        IFDOffset_uint32 = 8
        if swapbytes_flag:
            IFDOffset = numpy.uint32([IFDOffset_uint32]).byteswap().view(dtype=numpy.uint8)
        else:
            IFDOffset = numpy.uint32([IFDOffset_uint32]).view(dtype=numpy.uint8)
        IFDOffset.tofile(f)
        # endregion

        # region IFD common part
        ifd = Simple_IFD(endian)
        ifd.ImageWidth = width
        ifd.ImageLength = height
        ifd.RowsPerStrip = height
        ifd.set_dtype(data_type)
        ifd.StripByteCounts = (width * height * ifd.BitsPerSample // 8)
        ifd.NextIFD = 0
        # endregion

        # region Image description part
        hyperstack_flag = 0
        images = channels * slices * frames
        image_description = "ImageJ=1.53t\nimages={0:d}\n".format(images)
        if channels > 1:
            hyperstack_flag += 1
            image_description += "channels={0:d}\n".format(channels)
        if slices > 1:
            hyperstack_flag += 1
            image_description += "slices={0:d}\n".format(slices)
        if frames > 1:
            hyperstack_flag += 1
            image_description += "frames={0:d}\n".format(frames)
        if hyperstack_flag > 1:
            image_description += "hyperstack=true\n"
        if channels > 1:
            image_description += "mode=composite\n"
        else:
            image_description += "mode=grayscale\n"
        image_description += "unit=\\u00B5m\n"  # "\u00B5" is Unicode Character 'MICRO SIGN' (U+00B5)
        image_description += "spacing={0:.3f}\n".format(spacing)
        image_description += "loop=false\n"
        if (min is not None) and (max is not None):
            image_description += "min={0:.1f}\n".format(min)
            image_description += "max={0:.1f}\n".format(max)
            image_description_shift = len(
                bytes("min={0:.1f}\n".format(min) + "max={0:.1f}\n".format(max), encoding="utf-8"))
        image_description += "\x00"
        ifd.ImageDescription = image_description
        # Noted by Xuesong 10/30/2020:
        # If using encoding="utf-8", string "\u00B5" will be encoded into byte array b'unit=\x00\xc2\xb5m\n'. An abnormal b'\xc2' will occur.
        # Imagej will read unit as "Âµm" instead of "µm" because ImageJ is based on Unicode code point U+00B5 whereas UTF-8 (in literal) is \xc2\xb5
        # So here we use "\\u00B5" with encoding="utf-8" or use “\u00B5” with encoding="latin-1"
        image_description = bytes(image_description, encoding="utf-8")
        ifd.NumberCharsInImageDescription = len(image_description)
        ifd.OffsetOfImageDescription = IFDOffset_uint32 + ifd.bytes.nbytes
        # endregion

        # region Image resolution part
        image_resolution = numpy.zeros(8, dtype=numpy.uint8)
        # Although we use ifd.ResolutionUnit = 3 (Centimeter)
        # ImageJ reads unit from metadata instead of ifd.ResolutionUnit.

        # Old version
        # resolution_numerator = 1000000  # Convert mm into um.
        # resolution_denominator = resolution * 1000

        # New version: More consistent with ImageJ (Xuesong 04/19/2023)
        resolution_numerator = round(1000000 / resolution)  # Convert um into integer.
        resolution_denominator = 1000000  # denominator is always 1,000,000
        if swapbytes_flag:
            image_resolution[0:4] = numpy.uint32([resolution_numerator]).byteswap().view(dtype=numpy.uint8)
            image_resolution[4:8] = numpy.uint32([resolution_denominator]).byteswap().view(dtype=numpy.uint8)
        else:
            image_resolution[0:4] = numpy.uint32([resolution_numerator]).view(dtype=numpy.uint8)
            image_resolution[4:8] = numpy.uint32([resolution_denominator]).view(dtype=numpy.uint8)
        ifd.XResolution = ifd.OffsetOfImageDescription + ifd.NumberCharsInImageDescription
        ifd.YResolution = ifd.XResolution + 8
        ifd.resolution = round(resolution_denominator / resolution_numerator, 5)  # Unit: um / pixel
        # endregion

        # region Write IFD, ImageDescription and XYResolution into file
        if rewrite_flag:
            # image_description is "image_description_shift" bytes longer than 1st write_IFD()
            ifd.StripOffsets = ifd.YResolution + 8 + 100 - image_description_shift
        else:
            if (min is not None) and (max is not None):
                ifd.StripOffsets = ifd.YResolution + 8
            else:
                # Leave enough space for rewrite image_description
                ifd.StripOffsets = ifd.YResolution + 8 + 100
                zero_padding = numpy.zeros(100, dtype=numpy.uint8)

        ifd.bytes.tofile(f)
        f.write(image_description)
        image_resolution.tofile(f)
        image_resolution.tofile(f)

        if (rewrite_flag is False) and (min is None) and (max is None):
            zero_padding.tofile(f)

        f.flush()
        # endregion

    return ifd


def parse_tif(File_object, verbose=False):
    """
    Open a file, determine that it's a TIF by parsing its header, and
    read through the TIF's Image File Directories (IFDs) one at a time
    to determine the structure of the TIF.
    See:
     partners.adobe.com/public/developer/en/tiff/TIFF6.pdf
    for reference.
    """

    header = Simple_IFD()
    [next_ifd_offset, header.endian] = parse_header(File_object=File_object, verbose=verbose)
    header = parse_ifd(File_object=File_object, header=header, ifd_offset=next_ifd_offset, verbose=verbose)
    return header


def parse_header(File_object, verbose):
    """
    Read the 8 bytes at the start of a file to determine:
    1. Does the file seem to be a TIF?
    2. Is it little or big endian?
    3. What is the address of the first IFD?
    """
    header = get_bytes_from_file(File_object=File_object, offset=0, num_bytes=8)
    if verbose:
        print("Header: ", header)
    if header[0] == 73 and header[1] == 73 and header[2] == 42 and header[3] == 0:  # Little-endian
        endian = "little"
    elif header[0] == 77 and header[1] == 77 and header[2] == 0 and header[3] == 42:  # Big-endian
        endian = "big"
    else:
        raise UserWarning("Not a TIF file")

    ifd_offset = bytes_to_int(byte_array=header[4:8], endian=endian)
    if verbose:
        print("This file is a {0}-endian tif.".format(endian))
        print("The offset of the first IFD is at {0} bytes.".format(ifd_offset))
    return ifd_offset, endian


def parse_ifd(File_object, header, ifd_offset, verbose):
    """
    An IFD has:
     2-bytes to tell how many tags
     12 bytes per tag
     4 bytes to store the next IFD offset
    """
    num_tags = bytes_to_int(byte_array=get_bytes_from_file(File_object=File_object, offset=ifd_offset, num_bytes=2),
                            endian=header.endian)
    if verbose:
        print("IFD at offset {0} bytes with {1} tags:".format(ifd_offset, num_tags))
    header.NumDirEntries = num_tags
    ifd_bytes = get_bytes_from_file(File_object=File_object, offset=ifd_offset + 2, num_bytes=12 * num_tags + 4)
    entries = {}

    """
    The first IFD starts immediately after the summary metadata.
    Each IFD will contain the same set of TIFF tags, except for the first one in each file,
    which contains two ImageJ metadata tags, and two copies of the ImageDescription tag.
    One of these contains a string needed by ImageJ to recognize these files, and the other contains OME metadata.
    """
    ImageDescription_flag = 0
    for t in range(num_tags):
        entry = ifd_bytes[12 * t: 12 * (t + 1)]
        if verbose:
            print("   Entry #{0}:".format(t), end='')
            for e in entry:
                print(" {0},".format(e), end='')
            print()
        [tag_id, value] = interpret_ifd_entry(File_object=File_object, entry=entry, endian=header.endian,
                                              verbose=verbose)
        if value is not None:
            entries[tag_id] = value

        if hasattr(header, str(tag_id)):
            if tag_id == "ImageDescription" and ImageDescription_flag > 0:
                # If we find the 2nd actual "ImageDescription", overwrite the 1st "ImageDescription" into the "OMEXMLMetadata"
                header.NumberCharsInOMEXMLMetadata = header.NumberCharsInImageDescription
                header.OffsetOfOMEXMLMetadata = header.OffsetOfImageDescription
                header.OMEXMLMetadata = header.ImageDescription
            else:
                exec("header.{0} = entries['{1}']".format(tag_id, tag_id))

            if tag_id == "ImageDescription":
                header.NumberCharsInImageDescription = len(entries['ImageDescription'])
                header.OffsetOfImageDescription = bytes_to_int(byte_array=entry[8:12], endian=header.endian)
                header.ImageDescription = str(entries["ImageDescription"], encoding='utf-8', errors='ignore')
                ImageDescription_flag = ImageDescription_flag + 1
            elif tag_id == "XResolution":
                resolution_numerator = bytes_to_int(byte_array=header.XResolution[0:4], endian=header.endian)
                resolution_denominator = bytes_to_int(byte_array=header.XResolution[4:8], endian=header.endian)
                # New version: More consistent with ImageJ (Xuesong 04/19/2023)
                if resolution_denominator == 1:
                    # For some cases, tiff files do not follow unit in ImageDescription
                    header.resolution = round(10000 / resolution_numerator, 5)  # Convert cm into um
                    # Unit: um / pixel
                else:
                    header.resolution = round(resolution_denominator / resolution_numerator, 5)
                    # Unit: um / pixel
            elif tag_id == "IJMetadata":
                header.IJMetadata = []
                start_index = 0
                for IJMetadata_index in range(len(header.IJMetadataByteCounts)):
                    ByteCounts = header.IJMetadataByteCounts[IJMetadata_index]
                    try:
                        IJMetadata_str = str(value[start_index:start_index + ByteCounts], encoding="utf-16")
                        IJMetadata_json = json.loads(IJMetadata_str)
                        header.IJMetadata.append(IJMetadata_json)
                    except:
                        pass
                    start_index = start_index + ByteCounts
                if len(header.IJMetadata) == 1:
                    header.IJMetadata = header.IJMetadata[0]
                    # print(type(header.IJMetadata))  # <class 'dict'>
            elif tag_id == "MicroManagerMetadata":
                header.NumberCharsInMicroManagerMetadata = len(entries['MicroManagerMetadata'])
                header.OffsetOfOMicroManagerMetadata = bytes_to_int(byte_array=entry[8:12], endian=header.endian)
                MicroManagerMetadata_str = str(entries["MicroManagerMetadata"], encoding='utf-8', errors='ignore')

                # Trim the strange char(s) at the end of this char array
                end_index = MicroManagerMetadata_str.rfind("}")
                MicroManagerMetadata_str = MicroManagerMetadata_str[0:end_index + 1]
                header.MicroManagerMetadata = json.loads(MicroManagerMetadata_str)
        elif verbose:
            print("Simple_IFD class does not contain this Tiff tag: {0}".format(tag_id))

    """
    Try to find "images", "channels", "slices", "frames", "unit" and "spacing" from the (2nd) ImageDescription
    """
    if header.ImageDescription is not None:
        ImageDescription_list = header.ImageDescription.split('\n')

        # Check "channels"
        if any("channels" in list_value for list_value in ImageDescription_list):
            matching = [list_value for list_value in ImageDescription_list if "channels" in list_value]
            char_index = matching[0].find("channels=")
            if char_index != -1:
                header.channels = int(matching[0][char_index + 9:])

        # Check "slices"
        if any("slices" in list_value for list_value in ImageDescription_list):
            matching = [list_value for list_value in ImageDescription_list if "slices" in list_value]
            char_index = matching[0].find("slices=")
            if char_index != -1:
                header.slices = int(matching[0][char_index + 7:])

        # Check "frames"
        if any("frames" in list_value for list_value in ImageDescription_list):
            matching = [list_value for list_value in ImageDescription_list if "frames" in list_value]
            char_index = matching[0].find("frames=")
            if char_index != -1:
                header.frames = int(matching[0][char_index + 7:])

        # Check "unit"
        if any("unit" in list_value for list_value in ImageDescription_list):
            matching = [list_value for list_value in ImageDescription_list if "unit" in list_value]
            char_index = matching[0].find("unit=")
            if char_index != -1:
                unit = matching[0][char_index + 5:]
                if unit == "um" or "micron" or "\u00B5m":
                    header.unit = "um"
                else:
                    header.unit = unit

        # Check "spacing"
        if any("spacing" in list_value for list_value in ImageDescription_list):
            matching = [list_value for list_value in ImageDescription_list if "spacing" in list_value]
            char_index = matching[0].find("spacing=")
            if char_index != -1:
                header.spacing = float(matching[0][char_index + 8:])

        # Finally check "images"
        if any("images" in list_value for list_value in ImageDescription_list):
            matching = [list_value for list_value in ImageDescription_list if "images" in list_value]
            char_index = matching[0].find("images=")
            if char_index != -1:
                header.images = int(matching[0][char_index + 7:])
        elif (header.channels or header.slices or header.frames) is not None:
            header.images = 1
            if header.channels is not None:
                header.images *= header.channels
            if header.slices is not None:
                header.images *= header.slices
            if header.frames is not None:
                header.images *= header.frames

    return header


def interpret_ifd_entry(File_object, entry, endian, verbose):
    """
    Each IFD entry is stored in a binary format. Decode this to a python dict.
    """
    tag_id = bytes_to_int(byte_array=entry[0:2], endian=endian)
    tag_id_lookup = {
        254: 'NewSubFileType',
        256: 'ImageWidth',
        257: 'ImageLength',
        258: 'BitsPerSample',
        259: 'Compression',
        262: 'PhotometricInterpretation',
        270: 'ImageDescription',
        273: 'StripOffsets',
        277: 'SamplesPerPixel',
        278: 'RowsPerStrip',
        279: 'StripByteCounts',
        282: 'XResolution',
        283: 'YResolution',
        296: 'ResolutionUnit',
        339: 'SampleFormat',
        50838: 'IJMetadataByteCounts',
        50839: 'IJMetadata',
        51123: 'MicroManagerMetadata'}

    data_type = bytes_to_int(byte_array=entry[2:4], endian=endian)
    data_type_lookup = {
        1: ('BYTE', 1),
        2: ('ASCII', 1),
        3: ('SHORT', 2),
        4: ('LONG', 4),
        5: ('RATIONAL', 8),
        6: ('SBYTE', 1),
        7: ('UNDEFINED', 8),
        8: ('SSHORT', 2),
        9: ('SLONG', 4),
        10: ('SRATIONAL', 8),
        11: ('FLOAT', 4),
        12: ('DOUBLE', 8),
    }

    try:
        tag_id = tag_id_lookup[tag_id]

        try:
            data_type, bytes_per_count = data_type_lookup[data_type]
        except KeyError:
            raise UserWarning("Unknown data type in TIF tag: ", data_type)

        data_count = bytes_to_int(byte_array=entry[4:8], endian=endian)
        value_size_bytes = data_count * bytes_per_count
        if value_size_bytes <= 4:
            """
            The DataOffset directly encode the value
            """
            value = entry[8:8 + value_size_bytes]
        else:
            """
            The DataOffset encodes a pointer to the value 
            """
            offset = bytes_to_int(byte_array=entry[8:12], endian=endian)
            value = get_bytes_from_file(File_object=File_object, offset=offset, num_bytes=value_size_bytes)

        """
        We still haven't converted the value from bytes yet, but at least we
        got the correct bytes that encode the value.
        """
        if data_type == 'BYTE':
            if verbose:
                print("   {0}:".format(tag_id))
        elif data_type == 'ASCII':
            if verbose:
                content = str(value, encoding='utf-8', errors='ignore')
                print("   {0}:\n{1}".format(tag_id, content))
        elif data_type == 'SHORT':
            if data_count == 1:
                value = bytes_to_int(byte_array=value, endian=endian)
            else:
                typestr = ({'big': '>', 'little': '<'}[endian] +
                           {'BYTE': 'u1', 'SHORT': 'u2', 'LONG': 'u4'}[data_type])
                value = numpy.frombuffer(buffer=value, dtype=numpy.dtype(typestr))
            if verbose:
                print("   {0}: {1}".format(tag_id, value))
        elif data_type == 'LONG':
            if data_count == 1:
                value = bytes_to_int(byte_array=value, endian=endian)
            else:
                typestr = ({'big': '>', 'little': '<'}[endian] +
                           {'BYTE': 'u1', 'SHORT': 'u2', 'LONG': 'u4'}[data_type])
                value = numpy.frombuffer(buffer=value, dtype=numpy.dtype(typestr))
            if verbose:
                print("   {0}: {1}".format(tag_id, value))
        elif data_type == 'RATIONAL':
            if verbose:
                print("   {0}: {1}".format(tag_id, value))
        else:
            pass

    except KeyError:
        if verbose:
            value = bytes_to_int(byte_array=entry[8:12], endian=endian)
            print("   Unknown tag ID in TIF: {0} with value / offset: {1}".format(tag_id, value))
        value = []
    return tag_id, value


def get_bytes_from_file(File_object, offset, num_bytes):
    File_object.seek(offset, 0)
    return File_object.read(num_bytes)


def bytes_to_int(byte_array, endian):
    if endian == "little":
        return sum(c * 256 ** i for i, c in enumerate(byte_array))
    elif endian == "big":
        return sum(c * 256 ** (len(byte_array) - 1 - i) for i, c in enumerate(byte_array))
    else:
        raise UserWarning("'endian' must be either big or little")


class Simple_IFD:
    """
    See also: https://www.fileformat.info/format/tiff/egff.htm

    An Image File Directory (IFD) is a collection of information similar to a header, and it is used to describe the
    bitmapped data to which it is attached. Like a header, it contains information on the height, width, and depth of
    the image, the number of color planes, and the type of data compression used on the bitmapped data.

    One of the misconceptions about TIFF is that the information stored in the IFD tags is actually part of the TIFF
    header IFH. In fact, this information is often referred to as the "TIFF Header Information."  While other formats
    do store the type of information found in the IFD in the header, the TIFF header does not contain this
    information. Therefore, it is possible to think of the IFDs in a TIFF file as extensions of the TIFF file header.

    A TIFF file may contain any number of images, from zero on up:
        * Each image is considered to be a separate subfile (i.e., a bitmap) and has an IFD describing the bitmapped data.
        * Each TIFF subfile can be written as a separate TIFF file or can be stored with other subfiles in a single TIFF file.
        * Each subfile bitmap and IFD may reside anywhere in the TIFF file after the headers, and there may be only one IFD per image.
    The last field of every IFD contains an offset value to the next IFD, if any.
    If the offset value of any IFD is 00h, then there are no more images left to read in the TIFF file.

    The format of an Image File Directory (IFD) is shown in the following structure:
    typedef struct _TifIfd
    {
        WORD NumDirEntries; /* Number of Tags in IFD */
        TIFTAG TagList[]; /* Array of Tags */
        DWORD NextIFDOffset; /* Offset to next IFD */
    } TIFIFD;

    NumDirEntries: is a 2-byte value indicating the number of tags found in the IFD. Following this field is a
    series of tags; the number of tags corresponds to the value of the NumDirEntries field.

    TagList: Each tag structure is 12 bytes in size and, in the sample code above, is represented by an array of
    structures of the data type denition TIFTAG. The number of tags per IFD is limited to 65,535.

    NextIFDOffset: contains the offset position of the beginning of the next IFD. If there are no more IFDs,
    then the value of this field is 00h.
    """

    """
    Tags: a tag can be thought of as a data field in a file header.
    Difference: A header field used to hold a byte of data need only be a byte in size.
                A tag containing one byte of information, however, must always be twelve bytes in size.
    A TIFF tag has the following 12-byte structure:
    typedef struct _TifTag
    {
        WORD TagId; /* The tag identifier, 2 bytes */
        WORD DataType; /* The scalar type of the data items, 2 bytes */
        DWORD DataCount; /* The number of items in the tag data, 4 bytes */
        DWORD DataOffset; /* The byte offset to the data items, 4 bytes */
    } TIFTAG;

    TagId: is a numeric value identifying the type of information the tag contains.
        More specically, the TagId indicates what the tag information represents.
        Typical information found in every TIFF file includes:
            --- the height and width of the image, --- the depth of each pixel, and --- the type of data encoding used to compress the bitmap.
        Tags are normally identied by their TagId value and should always be written to an IFD in ascending order of the values found in the TagId field.

    DataType: contains a value indicating the scalar data type of the information found in the tag.
            The following values are supported:
                1 BYTE      8-bit unsigned integer;
                2 ASCII     8-bit, NULL-terminated string;
                3 SHORT     16-bit unsigned integer;
                4 LONG      32-bit unsigned integer;
                5 RATIONAL  Two 32-bit unsigned integers.
            The BYTE, SHORT, and LONG data types correspond to the BYTE, WORD, and DWORD data types used.
            The ASCII data type contains strings of 7-bit ASCII character data, which are always NULL-terminated and may be padded out to an even length if necessary.
            The RATIONAL data type is actually two LONG values and is used to store the two components of a fractional value.
            The 1st value stores the numerator, and the 2nd value stores the denominator.

    DataCount: indicates the number of items referenced by the tag and doesn't show the actual size of the data itself.
        Therefore, a DataCount of 08h does not necessarily indicate that eight bytes of data exist in the tag.
        This value indicates that eight items exist for the data type specied by this tag.
        For example, a DataCount value of 08h and a DataType of 03h indicate that the tag data is eight contiguous 16-bit unsigned integers, a total of 16 bytes in size.
            A DataCount of 28h and a DataType of 02h indicate an ASCII character string 40 bytes in length,
            including the NULL-terminator character, but not any padding if present.
            And a DataCount of 01h and a DataType of 05h indicate a single RATIONAL value a total of 8 bytes in size.

    DataOffset: is a 4-byte field that contains the offset location of the actual tag data within the TIFF file.
        1. If the tag data is 4 bytes or less in size, the data may be found in this field.
        2. If the tag data is greater than 4 bytes in size, then this field contains an offset to the position of the data in the TIFF file.
        Packing data within the DataOffset field is an optimization within the TIFF specication and is not required to be performed.
        Most data is typically stored outside the tag, occurring before or after the IFD.



    Note that several of these tags have default values that are used if the tag does not actually appear in a TIFF file.
    Bi-level (formerly Class B) and Gray-scale (formerly Class G) TIFF files must contain the 13 tags listed.
    These tags must appear in all revision 5.0 and 6.0 TIFF files regardless of the type of image data stored.
    Minimum Required Tags for TIFF Class B and Class G:
    Tag Type      TagId (2 bytes)       Tag Name                      DataType               DataCount N           default
      254              00feh            NewSubfileTyp            dword (LONG, 4 bytes)            1                   0
                  Currently defined values for the bitmap are:        0 - Image is reduced of another TIFF image in this file
                                                                      1 - Image is a single page of a multi-page
                                                                      2 - Image is a transparency mask for another image in this file

      256              0100h            ImageWidth               word or dword                    1               No default
                  The image's width, in pixels (X:horizontal). The number of columns in the image.

      257              0101h            ImageLength              word or dword                    1               No default
                  The image's length (height) in pixels (Y:vertical). The number of rows in the image.

      258              0102h            BitsPerSample            word (SHORT, 2 bytes)     SamplesPerPixel            1
                  Number of bits per sample (bit depth). Note that this tag allows a different number of bits per sample for each sample corresponding to a pixel.
                  For example, RGB color data could use a different number of bits per sample for each of the three color planes.

      259              0103h            Compression                    word                       1                   1
                  1 = No compression, but pack data into bytes as tightly as possible, with no unused bits except at the end of a row.
                      The byte ordering of data >8 bits must be consistent with that specified in the TIFF file header (bytes 0 and 1).
                      Rows are required to begin on byte boundaries.
                  2 = CCITT Group 3 1-Dimensional Modified Huffman run length encoding.
                  3 = Facsimile-compatible CCITT  Group 3.
                  4 = Facsimile-compatible CCITT  Group 4.
                  5 = LZW Compression, for grayscale, mapped color, and full color images.
                  32773 = PackBits compression, a simple byte oriented run length scheme for 1-bit images.

      262              0106h            PhotometricInterpretation      word                       1               No default
                  0 = For bilevel and grayscale images: 0 is imaged as white. 2**BitsPerSample-1 is imaged as black.
                      If GrayResponseCurve exists, it overrides the PhotometricInterpretation value.
                  1 = For bilevel and grayscale images: 0 is imaged as black. 2**BitsPerSample-1 is imaged as white.
                      If GrayResponseCurve exists, it overrides the PhotometricInterpretation value.
                  2 = RGB. In the RGB model, a color is described as a combination of the three primary colors of light (red, green, and blue) in particular concentrations.
                      For each of the three samples,  0 represents minimum intensity, and 2**BitsPerSample - 1 represents maximum intensity.
                      For PlanarConfiguration = 1, the samples are stored in the indicated order: first Red, then Green, then Blue.
                      For PlanarConfiguration = 2, the StripOffsets for the sample planes are stored in the indicated order:
                                                   first the Red sample plane StripOffsets, then the Green plane StripOffsets, then the Blue plane StripOffsets.
                  3 = "Palette color." In this mode, a color is described with a single sample.
                      The sample is used as an index into ColorMap. The sample is used to index into each of the red, green and blue curve tables to retrieve an RGB triplet defining an actual color.
                      When this PhotometricInterpretation value is used, the color response curves must also be supplied. SamplesPerPixel must be 1.
                  4 = Transparency Mask. This means that the image is used to define an irregularly shaped region of another image in the same TIFF file.
                      SamplesPerPixel and BitsPerSample must be 1. PackBits compression is recommended. The 1-bits define the interior of the region;
                      the 0-bits define the exterior of the region. The Transparency Mask must have the same ImageLength and ImageWidth as the main image.

      273              0111h            StripOffsets             word or dword    = StripsPerImage for PlanarConfiguration equal to 1;   No default
                                                                                  = SamplesPerPixel * StripsPerImage for PlanarConfiguration equal to 2.
                  For each strip, the byte offset of that strip. The offset is specified with respect to the beginning of the TIFF file.
                  Note that this implies that each strip has a location independent of the locations of other strips. This feature may be useful for editing applications.
                  This field is the only way for a reader to find the image data, and hence must exist.

      277              0115h            SamplesPerPixel                word                       1                   1
                  The number of samples per pixel. SamplesPerPixel is 1 for bilevel, grayscale, and palette color images.
                                                   SamplesPerPixel is 3 for RGB images.

      278              0116h            RowsPerStrip             word or dword                    1               2**32 - 1 (effectively infinity. That is, the entire image is one strip. Recomended is a strip size of 8K.)
                  The number of rows per strip.  The image data is organized into strips for fast access to individual rows when the data is compressed
                                               - though this field is valid even if the data is not compressed.
                  Noted be Xuesong 2019/04/16: the RowsPerStrip value species the maximum value, and not the required value, of the number of rows per strip.
                                               Many TIFF files, in fact, store a single strip of data and specify an arbitrarily large RowsPerStrip value.

      279              0117h            StripByteCounts          word or dword     = StripsPerImage for PlanarConfiguration equal to 1;   No default
                                                                                   = SamplesPerPixel * StripsPerImage for PlanarConfiguration equal to 2.
                  For each strip, the number of bytes in that strip. The existence of this field greatly
                                  simplifies the chore of buffering compressed data, if the strip size is reasonable.

      282              011Ah            XResolution                  RATIONAL                     1               No default
                  The number of pixels per ResolutionUnit in the X direction, i.e., in the  ImageWidth direction
      283              011Bh            YResolution                  RATIONAL                     1               No default
                  The number of pixels per ResolutionUnit in the Y direction, i.e., in the ImageLength direction.
      296              0128h            ResolutionUnit                 word                       1                    2
                  1 = No absolute unit of measurement. Used for images that may have a non-square aspect ratio, but no meaningful absolute dimensions.
                      The drawback of ResolutionUnit = 1 is that different applications will import the image at different sizes.
                      Even if the decision is quite arbitrary, it might be better to use dots per inch or dots per centimeter,
                      and pick XResolution and YResolution such that the aspect ratio is correct and the maximum dimension of the image is about four inches.
                  2 = Inch.
                  3 = Centimeter.

    Palette-color (formerly Class P) TIFF files add a 14th required tag that describes the type of palette information found within the TIFF image file:
      320              0140h            ColorMap

    RGB (formerly Class R) TIFF files contain the same tags as bi-level TIFF (Class B) files and add a 14th required tag, which describes the format of the bitmapped data in the image:
      284              011ch            PlanarConfiguration            word                       1                    1
                  1 = The sample values for each pixel are stored contiguously, so that there is a single image plane.
                      See PhotometricInterpretation to determine the order of the samples within the pixel data.
                      So, for RGB data, the data is stored RGBRGBRGB...and so on.
                  2 = The samples are stored in separate "sample planes."
                      The values in StripOffsets and StripByteCounts are then arranged as a 2-dimensional array, with SamplesPerPixel rows and StripsPerImage columns.
                      (All of the columns for row 0 are stored first, followed by the columns of row 1, and so on.)
                      PhotometricInterpretation describes the type of data that is stored in each sample plane.
                      For example,  RGB data is stored with the Red samples in one sample plane, the Green in another, and the Blue in another.
                  Noted by Xuesong 2019/04/16: If SamplesPerPixel is 1 (bilevel, grayscale),
                                               PlanarConfiguration is irrelevant, and should not be included.

    YCbCr TIFF files add 4 additional tags to the baseline:
      529              0211h            YCbCrCoecients
      530              0212h            YCbCrSubSampling
      531              0213h            YCbCrPositioning
      532              0214h            ReferenceBlackWhite
    
    
    In ImageJ, the following tags are also required:
      259              0103h            Compression           Default = 1, No compression
      270              010eh            ImageDescription              ASCII
                  In MicroManager, contains OME XML metadata (first IFD only)
      270              010eh            ImageDescription              ASCII            
                  For example, a user may wish to attach a comment such as "1988 company picnic" to an image.
                  (first IFD only)-–contains ImageJ file opening information.
    (Note: only MicroManager contains a 2nd ImageDescription tag.)  
    
      282              011Ah            XResolution           No default;
      283              011Bh            YResolution           No default; That is probably the reason why iSIM tiff data cannot be directed read by Windows. By Xuesong 2019/04/16
      296              0128h            ResolutionUnit        Default = 2, Inch
      
      339              0153h            SampleFormat             word (SHORT, 2 bytes)     SamplesPerPixel   1 (unsigned integer data)
                  Specifies how to interpret each data sample in a pixel. The specification defines these values:
                  1 = unsigned integer data;                  SAMPLEFORMAT_UINT = 1;
                  2 = two's complement signed integer data;   SAMPLEFORMAT_INT = 2;
                  3 = IEEE floating point data;               SAMPLEFORMAT_IEEEFP = 3;
                  4 = undefined data format.                  SAMPLEFORMAT_VOID = 4;
                  Noted by Xuesong 2019/04/16: SampleFormat field does not specify the size of data samples;
                                               this is still done by the BitsPerSample field.
    """

    """
    The following tags are only used in MicroManager. See more on: https://micro-manager.org/Micro-Manager_File_Formats
    
    Tag Type      TagId (2 bytes)       Tag Name                      DataType               DataCount N           default
      50838            c696h            IJMetadataByteCounts     dword (LONG, 4 bytes)
    IJMetadataByteCounts: TagId = 50838 (C6; 96); DataType = 4 (LONG); DataCount = 5; DataOffset = 0
    Little endian:   150, 198,   4,   0,   5,   0,   0,   0,   0,   0,   0,   0
    (first IFD only)
    
      50839            c697h            IJMetadata                     BYTE
    IJMetadata : TagId = 50839 (C6; 97); DataType = 1 (BYTE); DataCount = 1; DataOffset = 0
    Little endian:   151, 198,   1,   0,   1,   0,   0,   0,   0,   0,   0,   0
    (first IFD only)
    
      51123            c7b3h            MicroManagerMetadata          ASCII
    MicroManagerMetadata: TagId = 50839 (C6; 97); DataType = 2 (ASCII); DataCount = 1; DataOffset = 0
    Little endian:   179, 199,   2,   0,   1,   0,   0,   0,   0,   0,   0,   0
    
    Immediately after these tags are written:
        4 bytes containg the offset of the next IFD (per the TIFF specification)
        The pixel data
        In RGB files only, 6 bytes containing the values of the BitsPerSample tag Pixel values
        16 bytes containing the values of the XResolution and YResolution tags
        The value of the MicroManagerMetadata tag: image metadata (UTF-8 JSON)
    
    End of file:
        After the last IFD, the following constructs are written:
        
    ImageJ Metadata:
        A subset of the metadata used by the ImageJ TIFF writer (ij.io.TiffEncoder.java),
        which allows contrast settings and acquisition comments to propagate into ImageJ.
        The position and size of this metadata is specified by the IJMetadataCounts and IJMetadata tags in the first IFD.
        
    OME XML Metadata:
        A string containing the OME XML metadata for this data set.
        This String is referenced by the first of the two ImageDescription tags in the first IFD of the file, in accordance with the OME-TIFF specification.
        Since this String must be identical for all files in a data set, it is not written for any file until the entire data set is closed at the conclusion of an acquisition.
    
    ImageJ Image Description String:
        The ImageJ image description String that allows these files to opened correctly as hyperstacks in ImageJ.
        This String is referenced by the second of the two ImageDescription tags in the first IFD of the file.
    """

    def __init__(self, endian='little'):
        """
        A very simple TIF IFD with 15 tags (2 + 15*12 + 4 = 186 bytes)
        """
        self.NumDirEntries = 15
        self.NewSubFileType = 0
        self.ImageWidth = 0
        self.ImageLength = 0
        self.BitsPerSample = 0
        self.Compression = 1
        self.PhotometricInterpretation = 1

        # Image description part #2
        self.NumberCharsInImageDescription = 0
        self.OffsetOfImageDescription = 0
        self.ImageDescription = None

        self.StripOffsets = 0
        self.SamplesPerPixel = 1
        self.RowsPerStrip = 0
        self.StripByteCounts = 0
        self.XResolution = 0
        self.YResolution = 0
        self.ResolutionUnit = 3
        self.SampleFormat = 1

        # (ImageJ only)
        self.IJMetadataByteCounts = None
        self.IJMetadata = None

        # Image description part #1 (MicroManager only)
        self.NumberCharsInOMEXMLMetadata = 0
        self.OffsetOfOMEXMLMetadata = 0
        self.OMEXMLMetadata = None

        self.NumberCharsInMicroManagerMetadata = 0
        self.OffsetOfOMicroManagerMetadata = 0
        self.MicroManagerMetadata = None
        self.MicroManagerBigTiff = False

        self.NextIFD = 0
        self.endian = endian
        self.resolution = None
        self.images = None
        self.channels = None
        self.slices = None
        self.frames = None
        self.unit = None
        self.spacing = None

    @property
    def bytes(self):
        if sys.byteorder != self.endian:
            swapbytes_flag = True
        else:
            swapbytes_flag = False

        # NumDirEntries: Number of Tags in IFD (15, little endian byte ordering, maximum: 65535, "255, 255", "FF, FF")
        # Little endian:    [15, 0]
        # Big endian:       [0, 15]
        ifd = numpy.zeros(2, dtype=numpy.uint8, order='C')
        if swapbytes_flag:
            ifd[0:2] = numpy.uint16([self.NumDirEntries]).byteswap().view(dtype=numpy.uint8)
        else:
            ifd[0:2] = numpy.uint16([self.NumDirEntries]).view(dtype=numpy.uint8)

        # NewSubFileType: TagId = 254 (FE, 00); DataType = 4 (LONG); DataCount = 1; DataOffset = 0
        # Little endian:    254,   0,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       0,   254,   0,   4,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=254, DataType='LONG', DataCount=1, DataOffset=self.NewSubFileType,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # ImageWidth: TagId = 256 (00, 01); DataType = 4 (LONG); DataCount = 1; DataOffset = 0
        # Little endian:    0,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   0,   0,   4,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=256, DataType='LONG', DataCount=1, DataOffset=self.ImageWidth,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # ImageLength: TagId = 257 (01, 01); DataType = 4 (LONG); DataCount = 1; DataOffset = 0
        # Little endian:    1,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   1,   0,   4,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=257, DataType='LONG', DataCount=1, DataOffset=self.ImageLength,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # BitsPerSample: TagId = 258 (02; 01); DataType = 3 (SHORT); DataCount = 1; DataOffset = 0
        # Little endian:    2,   1,   3,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   2,   0,   3,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=258, DataType='SHORT', DataCount=1, DataOffset=self.BitsPerSample,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # Compression: TagId = 259 (03; 01); DataType = 3 (SHORT); DataCount = 1; DataOffset = 1 (No compression)
        # Little endian:    3,   1,   3,   0,   1,   0,   0,   0,   1,   0,   0,   0
        # Big endian:       1,   3,   0,   3,   0,   0,   0,   1,   0,   1,   0,   0
        tag_structure = self.tag_array(TagId=259, DataType='SHORT', DataCount=1, DataOffset=self.Compression,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # PhotometricInterpretation: TagId = 262 (06; 01); DataType = 3 (SHORT); DataCount = 1; DataOffset = 1 (0 is imaged as black. 2**BitsPerSample-1 is imaged as white)
        # Little endian:    6,   1,   3,   0,   1,   0,   0,   0,   1,   0,   0,   0
        #             % Big endian:       1,   6,   0,   3,   0,   0,   0,   1,   0,   1,   0,   0
        tag_structure = self.tag_array(TagId=262, DataType='SHORT', DataCount=1,
                                       DataOffset=self.PhotometricInterpretation, Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # ImageDescription: TagId = 270 (0E; 01); DataType = 2 (ASCII); DataCount = 0; DataOffset = 0
        # Little endian:    14,   1,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   14,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=270, DataType='ASCII', DataCount=self.NumberCharsInImageDescription,
                                       DataOffset=self.OffsetOfImageDescription, Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # StripOffsets: TagId = 273 (11; 01); DataType = 4 (LONG); DataCount = 1; DataOffset = 0
        # Little endian:    17,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   17,   0,   4,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=273, DataType='LONG', DataCount=1, DataOffset=self.StripOffsets,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # SamplesPerPixel: TagId = 277 (15; 01); DataType = 3 (SHORT); DataCount = 1; DataOffset = 1 (bilevel; grayscale; and palette color images)
        # Little endian:    21,   1,   3,   0,   1,   0,   0,   0,   1,   0,   0,   0
        # Big endian:       1,   21,   0,   3,   0,   0,   0,   1,   0,   1,   0,   0
        tag_structure = self.tag_array(TagId=277, DataType='SHORT', DataCount=1, DataOffset=self.SamplesPerPixel,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # RowsPerStrip: TagId = 278 (16; 01); DataType = 3 (SHORT); DataCount = 1; DataOffset = 0
        # Little endian:    22,   1,   3,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   22,   0,   3,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=278, DataType='SHORT', DataCount=1, DataOffset=self.RowsPerStrip,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # StripByteCounts: TagId = 279 (17; 01); DataType = 4 (LONG); DataCount = 1; DataOffset = 0
        # Little endian:    23,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   23,   0,   4,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=279, DataType='LONG', DataCount=1, DataOffset=self.StripByteCounts,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # XResolution: TagId = 282 (1A; 01); DataType = 5 (RATIONAL); DataCount = 1; DataOffset = 0
        # Little endian:    23,   1,   4,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   23,   0,   4,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=282, DataType='RATIONAL', DataCount=1, DataOffset=self.XResolution,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # YResolution: TagId = 283 (1B; 01); DataType = 5 (RATIONAL); DataCount = 1; DataOffset = 0
        # Little endian:    27,   1,   5,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   27,   0,   5,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=283, DataType='RATIONAL', DataCount=1, DataOffset=self.YResolution,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # ResolutionUnit: TagId = 296 (28; 01); DataType = 3 (SHORT); DataCount = 1; DataOffset = 3 (Centimeter)
        # Little endian:    40,   1,   3,   0,   1,   0,   0,   0,   3,   0,   0,   0
        # Big endian:       1,   40,   0,   3,   0,   0,   0,   1,   0,   3,   0,   0
        tag_structure = self.tag_array(TagId=296, DataType='SHORT', DataCount=1, DataOffset=self.ResolutionUnit,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        # SampleFormat: TagId = 339 (53; 01); DataType = 3 (SHORT); DataCount = 1; DataOffset = 0
        # Little endian:    83,   1,   3,   0,   1,   0,   0,   0,   0,   0,   0,   0
        # Big endian:       1,   83,   0,   3,   0,   0,   0,   1,   0,   0,   0,   0
        tag_structure = self.tag_array(TagId=339, DataType='SHORT', DataCount=1, DataOffset=self.SampleFormat,
                                       Swapbytes_Flag=swapbytes_flag)
        ifd = numpy.concatenate((ifd, tag_structure), axis=0)

        next_ifd_offset = numpy.zeros(4, dtype=numpy.uint8, order='C')
        if swapbytes_flag == 0:
            next_ifd_offset[0:4] = numpy.uint32([self.NextIFD]).view(dtype=numpy.uint8)
        else:
            next_ifd_offset[0:4] = numpy.uint32([self.NextIFD]).byteswap().view(dtype=numpy.uint8)
        ifd = numpy.concatenate((ifd, next_ifd_offset), axis=0)

        return ifd

    def tag_array(self, TagId, DataType, DataCount, DataOffset, Swapbytes_Flag):
        data_type_lookup = {
            'BYTE': (1, 1),
            'ASCII': (2, 1),
            'SHORT': (3, 2),
            'LONG': (4, 4),
            'RATIONAL': (5, 8),
            'SBYTE': (6, 1),
            'UNDEFINED': (7, 8),
            'SSHORT': (8, 2),
            'SLONG': (9, 4),
            'SRATIONAL': (10, 8),
            'FLOAT': (11, 4),
            'DOUBLE': (12, 8),
        }
        try:
            data_type, bytes_per_count = data_type_lookup[DataType]
        except KeyError:
            warning_string = "Unknown data type in TIF tag: %s" % DataType
            raise UserWarning(warning_string)
        value_size_bytes = DataCount * bytes_per_count

        entry = numpy.zeros(12, dtype=numpy.uint8, order='C')
        if Swapbytes_Flag:
            entry[0:2] = numpy.uint16([TagId]).byteswap().view(dtype=numpy.uint8)
            entry[2:4] = numpy.uint16([data_type]).byteswap().view(dtype=numpy.uint8)
            entry[4:8] = numpy.uint32([DataCount]).byteswap().view(dtype=numpy.uint8)
            if value_size_bytes == 1:
                entry[8] = DataOffset
            elif value_size_bytes == 2:
                entry[8:10] = numpy.uint16([DataOffset]).byteswap().view(dtype=numpy.uint8)
            else:
                entry[8:12] = numpy.uint32([DataOffset]).byteswap().view(dtype=numpy.uint8)
        else:
            entry[0:2] = numpy.uint16([TagId]).view(dtype=numpy.uint8)
            entry[2:4] = numpy.uint16([data_type]).view(dtype=numpy.uint8)
            entry[4:8] = numpy.uint32([DataCount]).view(dtype=numpy.uint8)
            if value_size_bytes == 1:
                entry[8] = DataOffset
            elif value_size_bytes == 2:
                entry[8:10] = numpy.uint16([DataOffset]).view(dtype=numpy.uint8)
            else:
                entry[8:12] = numpy.uint32([DataOffset]).view(dtype=numpy.uint8)
        return entry

    def set_dtype(self, dtype):
        allowed_dtypes = {
            # SampleFormat: 1 = unsigned integer data;
            numpy.dtype('uint8'): (1, 8),
            numpy.dtype('uint16'): (1, 16),
            numpy.dtype('uint32'): (1, 32),
            numpy.dtype('uint64'): (1, 64),
            # SampleFormat: 2 = two's complement signed integer data;
            numpy.dtype('int8'): (2, 8),
            numpy.dtype('int16'): (2, 16),
            numpy.dtype('int32'): (2, 32),
            numpy.dtype('int64'): (2, 64),
            # SampleFormat: 3 = IEEE floating point data;
            # numpy.dtype('float16'): (3, 16), #Not supported in older numpy?
            numpy.dtype('float32'): (3, 32),
            numpy.dtype('float64'): (3, 64),
        }
        try:
            self.SampleFormat, self.BitsPerSample = allowed_dtypes[dtype]
        except KeyError:
            warning_string = "Array datatype (%s) not allowed. Allowed types:" % (dtype)
            for i in sorted(allowed_dtypes.keys()):
                warning_string += "\n " + repr(i)
            raise UserWarning(warning_string)


if __name__ == '__main__':
    # region Write a 8-bit 2D tiff file using default settings (resolution = 0.05 um, spacing = 0.1 um)
    stack_in = numpy.random.randint(low=0, high=256, size=(512, 512), dtype='uint8')
    numpy.arange(500 * 1920 * 1600, dtype=numpy.float32).reshape([500, 1600, 1920])
    start_time = time.time()
    WriteTifStack(numpy_array=stack_in, filename='8_bit(default_settings).tif')
    end_time = time.time()
    print('Execution time of writing a 8-bit 2D tiff file: ', end_time - start_time)
    # endregion

    # region Read a 8-bit 2D tiff file
    start_time = time.time()
    stack_out = ReadTifStack(filename='8_bit(default_settings).tif')[0]
    end_time = time.time()
    print('Execution time of reading a 8-bit 2D tiff file: ', end_time - start_time)

    if numpy.array_equal(stack_in, stack_out):
        print("True")
    # endregion

    # region Write a 16-bit 2D tiff file using default settings (resolution = 0.05 um, spacing = 0.1 um)
    stack_in = numpy.random.randint(low=0, high=65536, size=(512, 512), dtype='uint16')
    start_time = time.time()
    WriteTifStack(numpy_array=stack_in, filename='16_bit(default_settings).tif')
    end_time = time.time()
    print('Execution time of writing a 16-bit 2D tiff file: ', end_time - start_time)
    # endregion

    # region Read a 16-bit 2D tiff file
    start_time = time.time()
    stack_out = ReadTifStack(filename='16_bit(default_settings).tif')[0]
    end_time = time.time()
    print('Execution time of reading a 16-bit 2D tiff file: ', end_time - start_time)

    if numpy.array_equal(stack_in, stack_out):
        print("True")
    # endregion

    # region Read a 16-bit 2D tiff file into 1D ndarray
    start_time = time.time()
    stack_out = ReadTifStack_1d(filename='16_bit(default_settings).tif')[0]
    end_time = time.time()
    print('Execution time of reading a 16-bit 2D tiff file into 1D ndarray: ', end_time - start_time)
    print(stack_out.shape)
    print(stack_out.dtype)
    # endregion

    # region Write a 32-bit 2D tiff file using default settings (resolution = 0.05 um, spacing = 0.1 um)
    stack_in = numpy.random.rand(512, 512)
    stack_in = stack_in.astype(dtype=numpy.dtype('float32'))
    start_time = time.time()
    WriteTifStack(numpy_array=stack_in, filename='32_bit(default_settings).tif')
    end_time = time.time()
    print('Execution time of writing a 32-bit 2D tiff file: ', end_time - start_time)
    # endregion

    # region Read a 32-bit 2D tiff file
    start_time = time.time()
    stack_out = ReadTifStack(filename='32_bit(default_settings).tif')[0]
    end_time = time.time()
    print('Execution time of reading a 32-bit 2D tiff file: ', end_time - start_time)

    if numpy.array_equal(stack_in, stack_out):
        print("True")
    # endregion

    # region Write a 8-bit 3D tiff file using user settings (resolution = 0.046 um, spacing = 0.2 um)
    stack_in = numpy.linspace(start=0, stop=256, num=(512 * 512), dtype='uint8')
    stack_in = numpy.tile(stack_in, 100)
    stack_in = stack_in.reshape([100, 512, 512])
    start_time = time.time()
    WriteTifStack(numpy_array=stack_in, filename='8_bit_stack(user_settings).tif', resolution=0.046, spacing=0.2)
    end_time = time.time()
    print('Execution time of writing a 8-bit 3D tiff file: ', end_time - start_time)
    # endregion

    # region Write a 16-bit 3D tiff file using user settings (resolution = 0.046 um, spacing = 0.2 um)
    stack_in = numpy.linspace(start=0, stop=65536, num=(1920 * 1280), dtype='uint16')
    stack_in = numpy.tile(stack_in, 100)
    stack_in = stack_in.reshape([100, 1280, 1920])
    start_time = time.time()
    WriteTifStack(numpy_array=stack_in, filename='16_bit_stack(user_setttings).tif', resolution=0.046, spacing=0.2)
    end_time = time.time()
    print('Execution time of writing a 16-bit 3D tiff file: ', end_time - start_time)
    # endregion

    # region Read a 16-bit 3D tiff file and display resolution, spacing and unit
    start_time = time.time()
    stack_out, header = ReadTifStack(filename='16_bit_stack(user_setttings).tif')
    end_time = time.time()
    print('Execution time of reading a 16-bit 3D tiff file: ', end_time - start_time)

    if numpy.array_equal(stack_in, stack_out):
        print("True")

    print("Resolution: ", header.resolution)
    print("Spacing: ", header.spacing)
    print("Unit: ", header.unit)
    # endregion

    # region Write a 16-bit 4D tiff file using user settings (resolution = 0.046 um, spacing = 0.2 um)
    stack_in = numpy.linspace(start=0, stop=65536, num=(1920 * 1280), dtype='uint16')
    stack_in = numpy.tile(stack_in, 100)
    stack_in = stack_in.reshape([25, 4, 1280, 1920])
    start_time = time.time()
    WriteTifStack(numpy_array=stack_in, filename='16_bit_4D_stack(user_setttings).tif', resolution=0.046, spacing=0.2,
                  imformat='zc')
    end_time = time.time()
    print('Execution time of writing a 16-bit 4D tiff file: ', end_time - start_time)
    # endregion

    # region Read a 16-bit 4D tiff file and display its channels, slices and frames
    start_time = time.time()
    stack_out, header = ReadTifStack(filename='16_bit_4D_stack(user_setttings).tif')
    end_time = time.time()
    print('Execution time of reading a 16-bit 4D tiff file: ', end_time - start_time)

    if numpy.array_equal(stack_in, stack_out):
        print("True")

    print("Channels: ", header.channels)
    print("Slices: ", header.slices)
    print("frames: ", header.frames)
    # endregion

    # region Write a 16-bit 5D tiff file using user settings and break the 4GB limit (resolution = 0.046 um, spacing = 0.2 um)
    stack_in = numpy.linspace(start=0, stop=65536, num=(1920 * 1280), dtype='uint16')
    stack_in = numpy.tile(stack_in, 2000)
    stack_in = stack_in.reshape([5, 100, 4, 1280, 1920])
    start_time = time.time()
    WriteTifStack(numpy_array=stack_in, filename='16_bit_5D_stack(tzc).tif', resolution=0.046, spacing=0.2,
                  imformat='tzc')
    end_time = time.time()
    print('Execution time of writing a 16-bit 5D tiff file: ', end_time - start_time)
    # endregion

    # region Read a 16-bit 5D tiff file
    start_time = time.time()
    stack_out, header = ReadTifStack(filename='16_bit_5D_stack(tzc).tif')
    end_time = time.time()
    print('Execution time of reading a 16-bit 5D tiff file: ', end_time - start_time)

    if numpy.array_equal(stack_in, stack_out):
        print("True")
    # endregion

    # region Get the header information about a 16-bit 5D tiff file without reading any image data
    start_time = time.time()
    header = get_header(filename='16_bit_5D_stack(tzc).tif')
    end_time = time.time()
    print('Execution time of reading a header: ', end_time - start_time)
    pprint(vars(header), indent=2)
    # endregion

    # region Compare with tifffile.imwrite
    start_time = time.time()
    tifffile.imwrite(
        '16_bit_5D_stack(tzc)_tifffile.tif',
        stack_in,
        imagej=True,
        resolution=(1. / 0.046, 1. / 0.046),
        metadata={
            'spacing': 0.2,
            'unit': 'um',
            'axes': 'TZCYX',
        }
    )
    end_time = time.time()
    print('Execution time of writing a 16-bit 5D ImageJ hyperstack formatted TIFF file using tifffile: ', end_time - start_time)
    # endregion

    # region Compare with tifffile.imread
    start_time = time.time()
    stack_out_tiff = tifffile.imread('16_bit_5D_stack(tzc).tif')
    end_time = time.time()
    print('Execution time of reading a 16-bit 5D tiff file using tifffile: ', end_time - start_time)

    if numpy.array_equal(stack_out, stack_out_tiff):
        print("True")
    # endregion

    # region Use tifffile.TiffFile to Read the volume and metadata from the Micro-Manager OME file
    start_time = time.time()
    with tifffile.TiffFile('Co-Local__1_MMStack_Default.ome.tif') as tif:
        stack_out = tif.asarray()
        axes = tif.series[0].axes
        imagej_metadata = tif.imagej_metadata
        ome_metadata = tif.ome_metadata
        micromanager_metadata = tif.micromanager_metadata
    end_time = time.time()
    print('Execution time of reading the volume and metadata from the 16-bit 5D ImageJ hyperstack file using tifffile: ', end_time - start_time)
    print(stack_out.shape)
    print("Axes: ", axes)
    pprint(imagej_metadata, indent=2)
    pprint(ome_metadata, indent=2)
    pprint(micromanager_metadata, indent=2)
    # endregion

    # # region Using "Append mode" to write a 16-bit 5D tiff file
    # stack_in = numpy.linspace(start=0, stop=65536, num=(1024 * 1024), dtype='uint16')
    # stack_in = numpy.tile(stack_in, 500)
    # stack_in = stack_in.reshape([5, 50, 2, 1024, 1024])
    #
    # width = stack_in.shape[4]       # 1024
    # height = stack_in.shape[3]      # 1024
    # channels = stack_in.shape[2]    # 2
    # slices = stack_in.shape[1]      # 50
    # frames = stack_in.shape[0]      # 5
    #
    # start_time = time.time()
    # write_IFD(filename='16_bit_5D_stack(tzc)_append.tif', endian='little', width=width, height=height,
    #           data_type=stack_in.dtype, channels=channels, slices=slices, frames=frames, spacing=0.2, resolution=0.046)
    # end_time = time.time()
    # print('Execution time of write_IFD: ', end_time - start_time)
    #
    # start_time = time.time()
    # with open(file='16_bit_5D_stack(tzc)_append.tif', mode='ab') as f:
    #     stack_in.tofile(f)
    # end_time = time.time()
    # print('Execution time of tofile: ', end_time - start_time)
    # # endregion
    #
    # # region Rewrite the IFD
    # start_time = time.time()
    # write_IFD(filename='16_bit_5D_stack(tzc)_append.tif', endian='little', width=width, height=height,
    #           data_type=stack_in.dtype, channels=channels, slices=slices, frames=frames, spacing=0.2,
    #           min=stack_in.min(), max=stack_in.max(), rewrite_flag=True)
    # end_time = time.time()
    # print('Execution time of re-write_IFD: ', end_time - start_time)
    # # endregion
    #
    # # region Read the 16-bit 5D tiff file written in "Append mode"
    # start_time = time.time()
    # stack_out = ReadTifStack(filename='16_bit_5D_stack(tzc)_append.tif')[0]
    # end_time = time.time()
    # print('Execution time of reading a 16-bit 5D tiff file of append mode: ', end_time - start_time)
    #
    # if numpy.array_equal(stack_in, stack_out):
    #     print("True")
    # # endregion

    # for i in range(500):
    #     data = numpy.random.randint(0, 65535, 1600 * 1920, dtype=numpy.uint16).reshape(1, 1600, 1920)
    #     tifffile.imwrite('temp_2.tif', data, append=True)




