package com.intel.analytics.zoo.common

import java.io.OutputStream
import java.nio.{ByteBuffer, ByteOrder}

import com.intel.analytics.bigdl.tensor.Tensor
import net.razorvine.pickle.{Opcodes, PickleUtils, Pickler}

class TensorPickler {
  def saveTensor(obj: Object, out: OutputStream): Unit = {
//    val jTensor = obj.asInstanceOf[Tensor]
//    val pickler = new Pickler(true)

  }

  private def saveBytes(out: OutputStream, pickler: Pickler, bytes: Array[Byte]): Unit = {
    out.write(Opcodes.BINSTRING)
    out.write(PickleUtils.integer_to_bytes(bytes.length))
    out.write(bytes)
  }

  private def floatArrayToBytes(arr: Array[Float]): Array[Byte] = {
    val bytes = new Array[Byte](4 * arr.size)
    val bb = ByteBuffer.wrap(bytes)
    bb.order(ByteOrder.nativeOrder())
    val db = bb.asFloatBuffer()
    db.put(arr)
    bytes
  }


}
