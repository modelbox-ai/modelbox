/*
 * Copyright 2021 The Modelbox Project Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


package com.modelbox;

/**
 * modelbox SessionContext
 */
public class SessionContext extends NativeObject {

  private SessionContext() {

  }

  /**
   * Set private object
   * @param key private key
   * @param object private object
   */
  public void setPrivate(String key, Object object) {
    SessionContext_SetPrivate(key, object);
  }

  /**
   * Get private object
   * @param key private key
   * @return object private object
   */
  public Object getPrivate(String key) {
    return SessionContext_GetPrivate(key);
  }

  /**
   * Set session ID
   * @param sessionId session id
   */
  public void setSessionId(String sessionId) {
    SessionContext_SetSessionId(sessionId);
  }

  /**
   * Get session ID
   * @return session id
   */
  public String getSessionId() {
    return SessionContext_GetSessionId();
  }

  /**
   * Get session configuration object
   * @return session configuration object
   */
  public Configuration getConfig() {
    return SessionContext_GetConfiguration();
  }

  /**
   * Set error to session
   * @param error flowunit error
   */
  public void setError(FlowUnitError error) {
    SessionContext_SetError(error);
  }

  /**
   * Get error from session
   * @return error flowunit error
   */
  public FlowUnitError getError() {
    return SessionContext_GetError();
  }

  private native void SessionContext_SetPrivate(String key, Object object);

  private native Object SessionContext_GetPrivate(String key);

  private native void SessionContext_SetSessionId(String sessionId);

  private native String SessionContext_GetSessionId();

  private native Configuration SessionContext_GetConfiguration();

  private native void SessionContext_SetError(FlowUnitError error);

  private native FlowUnitError SessionContext_GetError();
}
