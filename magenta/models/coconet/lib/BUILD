# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Description:
# Tools and models for using TensorFlow with music and art.

package(
    default_visibility = [
        "//magenta/models:__subpackages__",
    ],
)


licenses(["notice"])  # Apache 2.0

py_library(
    name = "lib",
    srcs = ["__init__.py"],
    deps = [
        ":data",
        ":evaluation",
        ":graph",
        ":hparams",
        ":logging",
        ":mask",
        ":pianoroll",
        ":sampling",
        ":tfutil",
        ":util",
    ]
)

py_library(
    name = "data",
    srcs = ["data.py"],
    data = [
        "testdata/Jsb16thSeparated.npz",
    ],
    deps = [
        ":mask",
        ":pianoroll",
        ":util",
        # tensorflow dep
  ]
)

py_library(
    name = "mask",
    srcs = ["mask.py"],
    deps = [
        ":util",
        # tensorflow dep
    ]
)


py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # numpy dep
    ]
)

py_library(
  name = "tfutil",
  srcs = ["tfutil.py"],
)

py_library(
  name = "pianoroll",
  srcs = ["pianoroll.py"],
)

py_library(
  name = "graph",
  srcs = ["graph.py"],
  deps = [
    ":util",
    ":tfutil",
  ]
)

py_library(
  name = "hparams",
  srcs = ["hparams.py"],
  deps = [
    ":util",
  ]
)


py_library(
    name = "logging",
    srcs = ["logging.py"],
)


py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    deps = [
        ":mask",
        ":data",
        ":util",
        ":tfutil",
        ":logging",
    ]
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    deps = [
        ":mask",
        ":util",
        ":tfutil",
    ]
)

