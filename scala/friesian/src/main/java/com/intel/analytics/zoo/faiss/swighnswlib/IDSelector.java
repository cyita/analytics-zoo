/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class IDSelector {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected IDSelector(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(IDSelector obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_IDSelector(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public boolean is_member(int id) {
    return swigfaissJNI.IDSelector_is_member(swigCPtr, this, id);
  }

}
