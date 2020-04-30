package com.intel.analytics.zoo.serving

import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.ipc.{ArrowFileReader, ArrowStreamWriter, WriteChannel}
import org.apache.arrow.vector.util.ByteArrayReadableSeekableByteChannel

import scala.collection.mutable.ArrayBuffer
import scala.util.control.Breaks._
import com.intel.analytics.bigdl.utils.T

import collection.JavaConverters._
import java.io.{ByteArrayOutputStream, IOException}
import java.nio.channels.Channel
import java.util

import org.apache.arrow.vector.{FieldVector, Float4Vector, VectorSchemaRoot}

object arrowtest {
  def main(args: Array[String]): Unit = {
    val alloc = new RootAllocator(Integer.MAX_VALUE)
    val MAX_ALLOC = 3 * 1024 * 1024 * 1024L
    val alloc4tensor = alloc.newChildAllocator("tensor", 0, MAX_ALLOC)

    // write tensor
    val floatVector = new Float4Vector("tensor", alloc4tensor)
    floatVector.setValueCount(10)
    val fields = ArrayBuffer(floatVector.getField).asJava
    val vectors = ArrayBuffer(floatVector.asInstanceOf[FieldVector]).asJava
    val root = new VectorSchemaRoot(fields, vectors,10)

    val out = new ByteArrayOutputStream()
//    val writer = new ArrowStreamWriter(root, null, new WriteChannel(out))

    val b =
    val reader = new ArrowFileReader(new ByteArrayReadableSeekableByteChannel(b), alloc4tensor)

    val dataList = new ArrayBuffer[Tensor[Float]]

    try {
      while (reader.loadNextBatch) {
        var shape = new ArrayBuffer[Int]()
        val vsr = reader.getVectorSchemaRoot
        val accessor = vsr.getVector("0")
        var idx = 0
        val valueCount = accessor.getValueCount
        breakable {
          while (idx < valueCount) {
            val data = accessor.getObject(idx).asInstanceOf[Float].toInt
            idx += 1
            if (data == Integer.MAX_VALUE) {
              break
            }
            shape += data
          }
        }
        val storage = new Array[Float](valueCount - idx)

        for (i <- idx until valueCount) {
          storage(i-idx) = accessor.getObject(i).asInstanceOf[Float]
        }

        val dataTensor = Tensor[Float](storage, shape.toArray)
        dataList += dataTensor
      }
    } catch {
      case ex: IOException =>
      // TODO
    } finally {
      reader.close()
    }

    if (dataList.isEmpty) {
      Tensor[Float]()
    } else if (dataList.length == 1) {
      dataList(0)
    } else {
      T.array(dataList.toArray)
    }
  }
}
