#
# Copyright 2016 The Analytics-Zoo Authors.
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

FROM ubuntu:18.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ENV JAVA_HOME                       /opt/jdk
ENV PATH                            ${JAVA_HOME}/bin:${PATH}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git && \
    apt-get install -y libsm6

#java
RUN wget http://enos.itcollege.ee/~jpoial/allalaadimised/jdk8/jdk-8u241-linux-x64.tar.gz && \
    gunzip jdk-8u241-linux-x64.tar.gz && \
    tar -xf jdk-8u241-linux-x64.tar -C /opt && \
    rm jdk-8u241-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_241 /opt/jdk

ADD analytics_zoo-0.9.0.dev0-py2.py3-none-manylinux1_x86_64.whl   /opt/work/

#python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install /opt/work/analytics_zoo-0.9.0.dev0-py2.py3-none-manylinux1_x86_64.whl && \
    pip install numpy scipy && \
    pip install --no-binary pandas -I pandas && \
    pip install scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests h5py opencv-python tensorflow==1.15.0 && \
    pip install install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install Pillow keras==2.0.8 cython scikit-image imgaug IPython[all] && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #Fix tornado await process
    pip uninstall -y -q tornado && \
    pip install tornado==5.1.1 && \
    python3 -m ipykernel.kernelspec

ADD BF16/  /opt/work/BF16_examples

CMD ["/bin/bash"]
