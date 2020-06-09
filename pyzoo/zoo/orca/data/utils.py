#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import numpy as np

from zoo.common import get_file_list


# split list into n chunks
def chunk(lst, n):
    size = len(lst) // n
    leftovers = lst[size * n:]
    for c in range(n):
        if leftovers:
            extra = [leftovers.pop()]
        else:
            extra = []
        yield lst[c * size:(c + 1) * size] + extra


def flatten(list_of_list):
    flattend = [item for sublist in list_of_list for item in sublist]
    return flattend


def list_s3_file(file_path, file_type, env):
    path_parts = file_path.split('/')
    bucket = path_parts.pop(0)
    key = "/".join(path_parts)

    access_key_id = env["AWS_ACCESS_KEY_ID"]
    secret_access_key = env["AWS_SECRET_ACCESS_KEY"]

    import boto3
    s3_client = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    ).client('s3', verify=False)
    # file
    if os.path.splitext(file_path)[1] != '':
        return ["s3://" + file_path]
    else:
        keys = []
        resp = s3_client.list_objects_v2(Bucket=bucket,
                                         Prefix=key)
        for obj in resp['Contents']:
            keys.append(obj['Key'])
        # only get json/csv files
        files = [file for file in keys if os.path.splitext(file)[1] == "." + file_type]
        file_paths = [os.path.join("s3://" + bucket, file) for file in files]
        return file_paths


def extract_one_path(file_path, file_type, env):
    file_url_splits = file_path.split("://")
    prefix = file_url_splits[0]
    if prefix == "s3":
        file_paths = list_s3_file(file_url_splits[1], file_type, env)
    elif prefix == "hdfs":
        import pyarrow as pa
        fs = pa.hdfs.connect()
        if fs.isfile(file_path):
            return [file_path]
        else:
            file_paths = get_file_list(file_path)
            # only get json/csv files
            file_paths = [file for file in file_paths
                          if os.path.splitext(file)[1] == "." + file_type]
    else:
        if os.path.isfile(file_path):
            return [file_path]
        else:
            file_paths = get_file_list(file_path)
            # only get json/csv files
            file_paths = [file for file in file_paths
                          if os.path.splitext(file)[1] == "." + file_type]
    return file_paths


def open_text(path):
    # Return a list of lines
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path, 'rb') as f:
            lines = f.read().decode("utf-8").split("\n")
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        lines = data["Body"].read().decode("utf-8").split("\n")
    else:  # Local path
        lines = []
        for line in open(path):
            lines.append(line)
    return [line.strip() for line in lines]


def open_image(path):
    from PIL import Image
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path, 'rb') as f:
            return Image.open(f)
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        from io import BytesIO
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(data["Body"].read()))
    else:  # Local path
        return Image.open(path)


def load_numpy(path):
    import numpy as np
    if path.startswith("hdfs"):  # hdfs://url:port/file_path
        import pyarrow as pa
        fs = pa.hdfs.connect()
        with fs.open(path, 'rb') as f:
            return np.load(f)
    elif path.startswith("s3"):  # s3://bucket/file_path
        access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
        secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]
        import boto3
        from io import BytesIO
        s3_client = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key).client('s3', verify=False)
        path_parts = path.split("://")[1].split('/')
        bucket = path_parts.pop(0)
        key = "/".join(path_parts)
        data = s3_client.get_object(Bucket=bucket, Key=key)
        return np.load(BytesIO(data["Body"].read()))
    else:  # Local path
        return np.load(path)


def check_type_and_convert(data, tuple_allowed=True, list_allowed=True):
    result = {}
    assert isinstance(data, dict), "each shard should be an dict"
    assert "x" in data, "key x should in each shard"
    x = data["x"]
    if isinstance(x, np.ndarray):
        new_x = [x]
    elif isinstance(x, tuple) and all([isinstance(xi, np.ndarray) for xi in x]):
        if tuple_allowed:
            new_x = x
        elif list_allowed:
            new_x = list(x)
        else:
            raise ValueError("value of x should be a ndarray")
    elif isinstance(x, list) and all([isinstance(xi, np.ndarray) for xi in x]):
        if list_allowed:
            new_x = x
        elif tuple_allowed:
            new_x = tuple(x)
        else:
            raise ValueError("value of x should be a ndarray")
    else:
        raise ValueError("value of x should be a ndarray, "
                         "a tuple of ndarrays or a list of ndarrays")
    result["x"] = new_x
    if "y" in data:
        y = data["y"]
        if isinstance(y, np.ndarray):
            new_y = [y]
        elif isinstance(y, tuple) and all([isinstance(yi, np.ndarray) for yi in y]):
            if tuple_allowed:
                new_y = y
            elif list_allowed:
                new_y = list(y)
            else:
                raise ValueError("value of y should be a ndarray")
        elif isinstance(y, list) and all([isinstance(yi, np.ndarray) for yi in y]):
            if list_allowed:
                new_y = y
            elif tuple_allowed:
                new_y = tuple(y)
            else:
                raise ValueError("value of y should be a ndarray")
        else:
            raise ValueError("value of y should be a ndarray, "
                             "a tuple of ndarrays or a list of ndarrays")
        result["y"] = new_y
    return result


def get_spec(tuple_allowed=True, list_allowed=True):
    def _get_spec(data):
        data = check_type_and_convert(data, tuple_allowed, list_allowed)
        feature_spec = [(feat.dtype, feat.shape[1:])
                        for feat in data["x"]]
        if "y" in data:
            label_spec = [(label.dtype, label.shape[1:])
                          for label in data["y"]]
        else:
            label_spec = None
        return feature_spec, label_spec
    return _get_spec


# todo this might be very slow
def flatten_xy(tuple_allowed=True, list_allowed=True):
    def _flatten_xy(data):
        data = check_type_and_convert(data, tuple_allowed, list_allowed)
        features = data["x"]

        has_label = "y" in data
        labels = data["y"] if has_label else None
        length = features[0].shape[0]

        for i in range(length):
            fs = [feat[i] for feat in features]
            if has_label:
                ls = [l[i] for l in labels]
                yield (fs, ls)
            else:
                yield (fs,)
    return _flatten_xy


def ray_partition_get_data_label(partition_data, tuple_allowed=True, list_allowed=True):
    from functools import reduce

    def combine_dict(dict1, dict2):
        return {key: np.concatenate((value, dict2[key]), axis=0)
                for (key, value) in dict1.items()}

    def combine_list_tuple(list1, list2):
        return [np.concatenate((list1[index], list2[index]), axis=0)
                for index in range(0, len(list1))]

    data_list = [data['x'] for data in partition_data]
    label_list = [data['y'] for data in partition_data]
    if isinstance(partition_data[0]['x'], dict):
        data = reduce(lambda dict1, dict2: combine_dict(dict1, dict2), data_list)
    elif isinstance(partition_data[0]['x'], np.ndarray):
        data = reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                      data_list)
    elif isinstance(partition_data[0]['x'], list):
        data = reduce(lambda list1, list2: combine_list_tuple(list1, list2), data_list)
        if not list_allowed and tuple_allowed:
            data = tuple(data)
        elif not list_allowed and not tuple_allowed:
            raise ValueError("value of x should be a ndarray")
    elif isinstance(partition_data[0]['x'], tuple):
        data = reduce(lambda tuple1, tuple2: combine_list_tuple(tuple1, tuple2), data_list)
        if not tuple_allowed and list_allowed:
            data = list(data)
        elif not list_allowed and not tuple_allowed:
            raise ValueError("value of x should be a ndarray")

    if isinstance(partition_data[0]['y'], dict):
        label = reduce(lambda dict1, dict2: combine_dict(dict1, dict2), label_list)
    elif isinstance(partition_data[0]['y'], np.ndarray):
        label = reduce(lambda array1, array2: np.concatenate((array1, array2), axis=0),
                       label_list)
    elif isinstance(partition_data[0]['y'], list):
        label = reduce(lambda list1, list2: combine_list_tuple(list1, list2), data_list)
        if not list_allowed and tuple_allowed:
            label = tuple(label)
        elif not list_allowed and not tuple_allowed:
            raise ValueError("value of y should be a ndarray")
    elif isinstance(partition_data[0]['y'], tuple):
        label = reduce(lambda tuple1, tuple2: combine_list_tuple(tuple1, tuple2), data_list)
        if not tuple_allowed and list_allowed:
            label = list(label)
        elif not list_allowed and not tuple_allowed:
            raise ValueError("value of y should be a ndarray")

    return data, label
