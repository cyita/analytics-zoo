/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.bigdl.friesian.serving.recall.faiss.swighnswlib;

public class IndexBinaryHNSW extends IndexBinary {
  private transient long swigCPtr;

  protected IndexBinaryHNSW(long cPtr, boolean cMemoryOwn) {
    super(swigfaissJNI.IndexBinaryHNSW_SWIGUpcast(cPtr), cMemoryOwn);
    swigCPtr = cPtr;
  }

  protected static long getCPtr(IndexBinaryHNSW obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_IndexBinaryHNSW(swigCPtr);
      }
      swigCPtr = 0;
    }
    super.delete();
  }

  public void setHnsw(HNSW value) {
    swigfaissJNI.IndexBinaryHNSW_hnsw_set(swigCPtr, this, HNSW.getCPtr(value), value);
  }

  public HNSW getHnsw() {
    long cPtr = swigfaissJNI.IndexBinaryHNSW_hnsw_get(swigCPtr, this);
    return (cPtr == 0) ? null : new HNSW(cPtr, false);
  }

  public void setOwn_fields(boolean value) {
    swigfaissJNI.IndexBinaryHNSW_own_fields_set(swigCPtr, this, value);
  }

  public boolean getOwn_fields() {
    return swigfaissJNI.IndexBinaryHNSW_own_fields_get(swigCPtr, this);
  }

  public void setStorage(IndexBinary value) {
    swigfaissJNI.IndexBinaryHNSW_storage_set(swigCPtr, this, getCPtr(value), value);
  }

  public IndexBinary getStorage() {
    long cPtr = swigfaissJNI.IndexBinaryHNSW_storage_get(swigCPtr, this);
    return (cPtr == 0) ? null : new IndexBinary(cPtr, false);
  }

  public IndexBinaryHNSW() {
    this(swigfaissJNI.new_IndexBinaryHNSW__SWIG_0(), true);
  }

  public IndexBinaryHNSW(int d, int M) {
    this(swigfaissJNI.new_IndexBinaryHNSW__SWIG_1(d, M), true);
  }

  public IndexBinaryHNSW(int d) {
    this(swigfaissJNI.new_IndexBinaryHNSW__SWIG_2(d), true);
  }

  public IndexBinaryHNSW(IndexBinary storage, int M) {
    this(swigfaissJNI.new_IndexBinaryHNSW__SWIG_3(getCPtr(storage), storage, M), true);
  }

  public IndexBinaryHNSW(IndexBinary storage) {
    this(swigfaissJNI.new_IndexBinaryHNSW__SWIG_4(getCPtr(storage), storage), true);
  }

  public DistanceComputer get_distance_computer() {
    long cPtr = swigfaissJNI.IndexBinaryHNSW_get_distance_computer(swigCPtr, this);
    return (cPtr == 0) ? null : new DistanceComputer(cPtr, true);
  }

  public void add(int n, SWIGTYPE_p_unsigned_char x) {
    swigfaissJNI.IndexBinaryHNSW_add(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x));
  }

  public void train(int n, SWIGTYPE_p_unsigned_char x) {
    swigfaissJNI.IndexBinaryHNSW_train(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x));
  }

  public void search(int n, SWIGTYPE_p_unsigned_char x, int k, SWIGTYPE_p_int distances, SWIGTYPE_p_long labels) {
    swigfaissJNI.IndexBinaryHNSW_search(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(x), k, SWIGTYPE_p_int.getCPtr(distances), SWIGTYPE_p_long.getCPtr(labels));
  }

  public void reconstruct(int key, SWIGTYPE_p_unsigned_char recons) {
    swigfaissJNI.IndexBinaryHNSW_reconstruct(swigCPtr, this, key, SWIGTYPE_p_unsigned_char.getCPtr(recons));
  }

  public void reset() {
    swigfaissJNI.IndexBinaryHNSW_reset(swigCPtr, this);
  }

}
