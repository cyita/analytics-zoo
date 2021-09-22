/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class BitstringWriter {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected BitstringWriter(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(BitstringWriter obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_BitstringWriter(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setCode(SWIGTYPE_p_unsigned_char value) {
    swigfaissJNI.BitstringWriter_code_set(swigCPtr, this, SWIGTYPE_p_unsigned_char.getCPtr(value));
  }

  public SWIGTYPE_p_unsigned_char getCode() {
    long cPtr = swigfaissJNI.BitstringWriter_code_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_unsigned_char(cPtr, false);
  }

  public void setCode_size(long value) {
    swigfaissJNI.BitstringWriter_code_size_set(swigCPtr, this, value);
  }

  public long getCode_size() {
    return swigfaissJNI.BitstringWriter_code_size_get(swigCPtr, this);
  }

  public void setI(long value) {
    swigfaissJNI.BitstringWriter_i_set(swigCPtr, this, value);
  }

  public long getI() {
    return swigfaissJNI.BitstringWriter_i_get(swigCPtr, this);
  }

  public BitstringWriter(SWIGTYPE_p_unsigned_char code, int code_size) {
    this(swigfaissJNI.new_BitstringWriter(SWIGTYPE_p_unsigned_char.getCPtr(code), code_size), true);
  }

  public void write(long x, int nbit) {
    swigfaissJNI.BitstringWriter_write(swigCPtr, this, x, nbit);
  }

}
