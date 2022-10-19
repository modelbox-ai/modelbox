package com.modelbox;

/**
 * modelbox JNI backend
 */
public class NativeObject {
  private long native_handle = 0;

  static {
    System.loadLibrary("modelbox-jni");
  }

  protected NativeObject() {
    native_handle = 0;
  }

  /**
   * Get native jni handle
   * @return native jni handle
   */
  public long getNativeHandle() {
    return native_handle;
  }

  /**
   * Set native jni handle
   * @param handle native jni handle
   */
  protected void setNativeHandle(long handle) {
    if (native_handle != 0) {
      delete_handle(native_handle);
      native_handle = 0;
    }
    native_handle = handle;
  }

  /**
   * Free native jni handle
   */
  @Override
  @SuppressWarnings("deprecation")
  protected void finalize() {
    try {
      delete_handle(native_handle);
      native_handle = 0;
    } catch (Exception e) {
      //pass
    }
  }

  private native void delete_handle(long handle);
}
