import argparse
import h5py
import json
import numpy as np
import copy


def main(args):
    h5_orig = h5py.File(args.input_file_name, "r")
    h5_new = h5py.File(args.output_file_name, "w")
    total_samples = 0
    for key in h5_orig["mask"].keys():
        array = h5_orig["mask"][key][:]
        array = copy.deepcopy([a for a in array if int(a[5:]) <= args.num_to_copy])
        if "mask" in h5_new and key in h5_new["mask"]:
            del h5_new[f"mask/{key}"]
        h5_new[f"mask/{key}"] = np.array(array, dtype="S")
    for i in range(1, args.num_to_copy + 1):
        h5_orig.copy(h5_orig[f"/data/demo_{i}/"], h5_new["/data/"])
        total_samples += h5_new[f"/data/demo_{i}/actions"].shape[0]
    env_meta = json.loads(h5_orig["data"].attrs["env_args"])
    h5_new["data"].attrs["env_args"] = json.dumps(env_meta)
    h5_new["data"].attrs["total"] = total_samples
    h5_orig.close()
    h5_new.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice hdf5 dataset.")
    parser.add_argument("--input_file_name", default="", help="input file name")
    parser.add_argument("--output_file_name", default="", help="output file name")
    parser.add_argument(
        "--num_to_copy", type=int, default=5000, help="output file name"
    )
    args = parser.parse_args()
    main(args)
