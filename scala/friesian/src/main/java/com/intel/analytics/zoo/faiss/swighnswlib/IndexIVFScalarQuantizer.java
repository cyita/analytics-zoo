/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class IndexIVFScalarQuantizer extends IndexIVF {
  private transient long swigCPtr;

  protected IndexIVFScalarQuantizer(long cPtr, boolean cMemoryOwn) {
    super(swigfaissJNI.IndexIVFScalarQuantizer_SWIGUpcast(cPtr), cMemoryOwn);
    swigCPtr = cPtr;
  }

  protected static long getCPtr(IndexIVFScalarQuantizer obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_IndexIVFScalarQuantizer(swigCPtr);
      }
      swigCPtr = 0;
    }
    super.delete();
  }

  public void setSq(ScalarQuantizer value) {
    swigfaissJNI.IndexIVFScalarQuantizer_sq_set(swigCPtr, this, ScalarQuantizer.getCPtr(value), value);
  }

  public ScalarQuantizer getSq() {
    long cPtr = swigfaissJNI.IndexIVFScalarQuantizer_sq_get(swigCPtr, this);
    return (cPtr == 0) ? null : new ScalarQuantizer(cPtr, false);
  }

  public void setBy_residual(boolean value) {
    swigfaissJNI.IndexIVFScalarQuantizer_by_residual_set(swigCPtr, this, value);
  }

  public boolean getBy_residual() {
    return swigfaissJNI.IndexIVFScalarQuantizer_by_residual_get(swigCPtr, this);
  }

  public IndexIVFScalarQuantizer(Index quantizer, long d, long nlist, ScalarQuantizer.QuantizerType qtype, MetricType metric, boolean encode_residual) {
    this(swigfaissJNI.new_IndexIVFScalarQuantizer__SWIG_0(Index.getCPtr(quantizer), quantizer, d, nlist, qtype.swigValue(), metric.swigValue(), encode_residual), true);
  }

  public IndexIVFScalarQuantizer(Index quantizer, long d, long nlist, ScalarQuantizer.QuantizerType qtype, MetricType metric) {
    this(swigfaissJNI.new_IndexIVFScalarQuantizer__SWIG_1(Index.getCPtr(quantizer), quantizer, d, nlist, qtype.swigValue(), metric.swigValue()), true);
  }

  public IndexIVFScalarQuantizer(Index quantizer, long d, long nlist, ScalarQuantizer.QuantizerType qtype) {
    this(swigfaissJNI.new_IndexIVFScalarQuantizer__SWIG_2(Index.getCPtr(quantizer), quantizer, d, nlist, qtype.swigValue()), true);
  }

  public IndexIVFScalarQuantizer() {
    this(swigfaissJNI.new_IndexIVFScalarQuantizer__SWIG_3(), true);
  }

  public void train_residual(int n, SWIGTYPE_p_float x) {
    swigfaissJNI.IndexIVFScalarQuantizer_train_residual(swigCPtr, this, n, SWIGTYPE_p_float.getCPtr(x));
  }

  public void encode_vectors(int n, SWIGTYPE_p_float x, SWIGTYPE_p_long list_nos, SWIGTYPE_p_unsigned_char codes, boolean include_listnos) {
    swigfaissJNI.IndexIVFScalarQuantizer_encode_vectors__SWIG_0(swigCPtr, this, n, SWIGTYPE_p_float.getCPtr(x), SWIGTYPE_p_long.getCPtr(list_nos), SWIGTYPE_p_unsigned_char.getCPtr(codes), include_listnos);
  }

  public void encode_vectors(int n, SWIGTYPE_p_float x, SWIGTYPE_p_long list_nos, SWIGTYPE_p_unsigned_char codes) {
    swigfaissJNI.IndexIVFScalarQuantizer_encode_vectors__SWIG_1(swigCPtr, this, n, SWIGTYPE_p_float.getCPtr(x), SWIGTYPE_p_long.getCPtr(list_nos), SWIGTYPE_p_unsigned_char.getCPtr(codes));
  }

  public void add_with_ids(int n, SWIGTYPE_p_float x, SWIGTYPE_p_long xids) {
    swigfaissJNI.IndexIVFScalarQuantizer_add_with_ids(swigCPtr, this, n, SWIGTYPE_p_float.getCPtr(x), SWIGTYPE_p_long.getCPtr(xids));
  }

  public SWIGTYPE_p_faiss__InvertedListScanner get_InvertedListScanner(boolean store_pairs) {
    long cPtr = swigfaissJNI.IndexIVFScalarQuantizer_get_InvertedListScanner(swigCPtr, this, store_pairs);
    return (cPtr == 0) ? null : new SWIGTYPE_p_faiss__InvertedListScanner(cPtr, false);
  }

  public void reconstruct_from_offset(int list_no, int offset, SWIGTYPE_p_float recons) {
    swigfaissJNI.IndexIVFScalarQuantizer_reconstruct_from_offset(swigCPtr, this, list_no, offset, SWIGTYPE_p_float.getCPtr(recons));
  }

  public void sa_decode(int n, SWIGTYPE_p_unsigned_char bytes, SWIGTYPE_p_float x) {
    swigfaissJNI.IndexIVFScalarQuantizer_sa_decode(swigCPtr, this, n, SWIGTYPE_p_unsigned_char.getCPtr(bytes), SWIGTYPE_p_float.getCPtr(x));
  }

}
