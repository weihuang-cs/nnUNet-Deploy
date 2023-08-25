import traceback
from typing import Type

import predictor
from predictor.common.file_and_folder_operations import join
from predictor.data_io.base_reader_writer import BaseReaderWriter
from predictor.data_io.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from predictor.data_io.simpleitk_reader_writer import SimpleITKIO
from predictor.common.utils import recursive_find_python_class

LIST_OF_IO_CLASSES = [SimpleITKIO, NibabelIO, NibabelIOWithReorient]


def determine_reader_writer_from_dataset_json(
    dataset_json_content: dict,
    example_file: str = None,
    allow_nonmatching_filename: bool = False,
    verbose: bool = True,
) -> Type[BaseReaderWriter]:
    if (
        "overwrite_image_reader_writer" in dataset_json_content.keys()
        and dataset_json_content["overwrite_image_reader_writer"] != "None"
    ):
        ioclass_name = dataset_json_content["overwrite_image_reader_writer"]
        # trying to find that class in the predictor.imageio module
        try:
            ret = recursive_find_reader_writer_by_name(ioclass_name)
            if verbose:
                print("Using %s reader/writer" % ret)
            return ret
        except RuntimeError:
            if verbose:
                print(
                    "Warning: Unable to find ioclass specified in dataset.json: %s"
                    % ioclass_name
                )
            if verbose:
                print("Trying to automatically determine desired class")
    return determine_reader_writer_from_file_ending(
        dataset_json_content["file_ending"],
        example_file,
        allow_nonmatching_filename,
        verbose,
    )


def determine_reader_writer_from_file_ending(
    file_ending: str,
    example_file: str = None,
    allow_nonmatching_filename: bool = False,
    verbose: bool = True,
):
    for rw in LIST_OF_IO_CLASSES:
        if file_ending.lower() in rw.supported_file_endings:
            if example_file is not None:
                # if an example file is provided, try if we can actually read it. If not move on to the next reader
                try:
                    tmp = rw()
                    _ = tmp.read_images((example_file,))
                    if verbose:
                        print("Using %s as reader/writer" % rw)
                    return rw
                except:
                    if verbose:
                        print(f"Failed to open file {example_file} with reader {rw}:")
                    traceback.print_exc()
                    pass
            else:
                if verbose:
                    print("Using %s as reader/writer" % rw)
                return rw
        else:
            if allow_nonmatching_filename and example_file is not None:
                try:
                    tmp = rw()
                    _ = tmp.read_images((example_file,))
                    if verbose:
                        print("Using %s as reader/writer" % rw)
                    return rw
                except:
                    if verbose:
                        print(f"Failed to open file {example_file} with reader {rw}:")
                    if verbose:
                        traceback.print_exc()
                    pass
    raise RuntimeError(
        "Unable to determine a reader for file ending %s and file %s (file None means no file provided)."
        % (file_ending, example_file)
    )


def recursive_find_reader_writer_by_name(rw_class_name: str) -> Type[BaseReaderWriter]:
    ret = recursive_find_python_class(
        join(predictor.__path__[0], "data_io"), rw_class_name, "predictor.data_io"
    )
    if ret is None:
        raise RuntimeError(
            "Unable to find reader writer class '%s'. Please make sure this class is located in the "
            "predictor.data_io module." % rw_class_name
        )
    else:
        return ret
