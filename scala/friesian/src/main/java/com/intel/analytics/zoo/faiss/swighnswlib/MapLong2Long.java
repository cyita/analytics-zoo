/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 3.0.12
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

package com.intel.analytics.zoo.faiss.swighnswlib;

public class MapLong2Long {
  private transient long swigCPtr;
  protected transient boolean swigCMemOwn;

  protected MapLong2Long(long cPtr, boolean cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = cPtr;
  }

  protected static long getCPtr(MapLong2Long obj) {
    return (obj == null) ? 0 : obj.swigCPtr;
  }

  protected void finalize() {
    delete();
  }

  public synchronized void delete() {
    if (swigCPtr != 0) {
      if (swigCMemOwn) {
        swigCMemOwn = false;
        swigfaissJNI.delete_MapLong2Long(swigCPtr);
      }
      swigCPtr = 0;
    }
  }

  public void setMap(SWIGTYPE_p_std__unordered_mapT_long_long_t value) {
    swigfaissJNI.MapLong2Long_map_set(swigCPtr, this, SWIGTYPE_p_std__unordered_mapT_long_long_t.getCPtr(value));
  }

  public SWIGTYPE_p_std__unordered_mapT_long_long_t getMap() {
    return new SWIGTYPE_p_std__unordered_mapT_long_long_t(swigfaissJNI.MapLong2Long_map_get(swigCPtr, this), true);
  }

  public void add(long n, SWIGTYPE_p_long keys, SWIGTYPE_p_long vals) {
    swigfaissJNI.MapLong2Long_add(swigCPtr, this, n, SWIGTYPE_p_long.getCPtr(keys), SWIGTYPE_p_long.getCPtr(vals));
  }

  public int search(int key) {
    return swigfaissJNI.MapLong2Long_search(swigCPtr, this, key);
  }

  public void search_multiple(long n, SWIGTYPE_p_long keys, SWIGTYPE_p_long vals) {
    swigfaissJNI.MapLong2Long_search_multiple(swigCPtr, this, n, SWIGTYPE_p_long.getCPtr(keys), SWIGTYPE_p_long.getCPtr(vals));
  }

  public MapLong2Long() {
    this(swigfaissJNI.new_MapLong2Long(), true);
  }

}
