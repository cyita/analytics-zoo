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

ARG ANALYTICS_ZOO_VERSION=0.9.0-SNAPSHOT
ARG BIGDL_VERSION=0.10.0
ARG SPARK_VERSION=2.4.3
ENV ANALYTICS_ZOO_VERSION           ${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION                   ${SPARK_VERSION}
ENV BIGDL_VERSION                   ${BIGDL_VERSION}
ENV SPARK_HOME                      /opt/work/spark-${SPARK_VERSION}
ENV ANALYTICS_ZOO_HOME              /opt/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME                       /opt/jdk
ENV PATH                            ${JAVA_HOME}/bin:${PATH}
ENV PYTHONPATH                      ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip:${PYTHONPATH}
ENV BIGDL_CLASSPATH                 ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git && \
    apt-get install -y libsm6

#java
RUN wget http://enos.itcollege.ee/~jpoial/allalaadimised/jdk8/jdk-8u241-linux-x64.tar.gz && \
    gunzip jdk-8u241-linux-x64.tar.gz && \
    tar -xf jdk-8u241-linux-x64.tar -C /opt && \
    rm jdk-8u241-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_241 /opt/jdk

#python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install numpy scipy && \
    pip install --no-binary pandas -I pandas && \
    pip install scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests h5py opencv-python tensorflow==1.15.0 && \
    pip install install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #Fix tornado await process
    pip uninstall -y -q tornado && \
    pip install tornado==5.1.1 && \
    python3 -m ipykernel.kernelspec

#spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz

ADD analytics-zoo-0.9.0-SNAPSHOT/   /opt/work/analytics-zoo-0.9.0-SNAPSHOT
ADD BF16_examples/  /opt/work/BF16_examples

CMD ["/bin/bash"]
