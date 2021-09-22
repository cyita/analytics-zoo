/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class BufferList {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected BufferList(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(BufferList obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_BufferList(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setBuffer_size(long value) {
    swigfaissJNI.BufferList_buffer_size_set(swigCPtr, this, value);
  }

  public long getBuffer_size() {
    return swigfaissJNI.BufferList_buffer_size_get(swigCPtr, this);
  }

  public void setBuffers(SWIGTYPE_p_std__vectorT_faiss__BufferList__Buffer_t value) {
    swigfaissJNI.BufferList_buffers_set(swigCPtr, this, SWIGTYPE_p_std__vectorT_faiss__BufferList__Buffer_t.getCPtr(value));
  }

  public SWIGTYPE_p_std__vectorT_faiss__BufferList__Buffer_t getBuffers() {
    long cPtr = swigfaissJNI.BufferList_buffers_get(swigCPtr, this);
    return (cPtr == 0) ? null : new SWIGTYPE_p_std__vectorT_faiss__BufferList__Buffer_t(cPtr, false);
  }

  public void setWp(long value) {
    swigfaissJNI.BufferList_wp_set(swigCPtr, this, value);
  }

  public long getWp() {
    return swigfaissJNI.BufferList_wp_get(swigCPtr, this);
  }

  public BufferList(long buffer_size) {
    this(swigfaissJNI.new_BufferList(buffer_size), true);
  }

  public void append_buffer() {
    swigfaissJNI.BufferList_append_buffer(swigCPtr, this);
  }

  public void add(int id, float dis) {
    swigfaissJNI.BufferList_add(swigCPtr, this, id, dis);
  }

  public void copy_range(long ofs, long n, SWIGTYPE_p_long dest_ids, SWIGTYPE_p_float dest_dis) {
    swigfaissJNI.BufferList_copy_range(swigCPtr, this, ofs, n, SWIGTYPE_p_long.getCPtr(dest_ids), SWIGTYPE_p_float.getCPtr(dest_dis));
  }

}
