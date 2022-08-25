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

import java.io.IOException;

/**
 * modelbox exception type
 */
public class ModelBoxException extends IOException {
  public ModelBoxException() {}

  public ModelBoxException(String message) {
    super(message);
  }

  static public class Success extends ModelBoxException {
    public Success() {
      super();
    }

    public Success(String message) {
      super(message);
    }
  }
  static public class Fault extends ModelBoxException {
    public Fault() {
      super();
    }

    public Fault(String message) {
      super(message);
    }
  }
  static public class Notfound extends ModelBoxException {
    public Notfound() {
      super();
    }

    public Notfound(String message) {
      super(message);
    }
  }
  static public class Invalid extends ModelBoxException {
    public Invalid() {
      super();
    }

    public Invalid(String message) {
      super(message);
    }
  }
  static public class Again extends ModelBoxException {
    public Again() {
      super();
    }

    public Again(String message) {
      super(message);
    }
  }
  static public class Badconf extends ModelBoxException {
    public Badconf() {
      super();
    }

    public Badconf(String message) {
      super(message);
    }
  }
  static public class Nomem extends ModelBoxException {
    public Nomem() {
      super();
    }

    public Nomem(String message) {
      super(message);
    }
  }
  static public class Range extends ModelBoxException {
    public Range() {
      super();
    }

    public Range(String message) {
      super(message);
    }
  }
  static public class Exist extends ModelBoxException {
    public Exist() {
      super();
    }

    public Exist(String message) {
      super(message);
    }
  }
  static public class Internal extends ModelBoxException {
    public Internal() {
      super();
    }

    public Internal(String message) {
      super(message);
    }
  }
  static public class Busy extends ModelBoxException {
    public Busy() {
      super();
    }

    public Busy(String message) {
      super(message);
    }
  }
  static public class Permit extends ModelBoxException {
    public Permit() {
      super();
    }

    public Permit(String message) {
      super(message);
    }
  }
  static public class Notsupport extends ModelBoxException {
    public Notsupport() {
      super();
    }

    public Notsupport(String message) {
      super(message);
    }
  }
  static public class Nodata extends ModelBoxException {
    public Nodata() {
      super();
    }

    public Nodata(String message) {
      super(message);
    }
  }
  static public class Nospace extends ModelBoxException {
    public Nospace() {
      super();
    }

    public Nospace(String message) {
      super(message);
    }
  }
  static public class Nobufs extends ModelBoxException {
    public Nobufs() {
      super();
    }

    public Nobufs(String message) {
      super(message);
    }
  }
  static public class Overflow extends ModelBoxException {
    public Overflow() {
      super();
    }

    public Overflow(String message) {
      super(message);
    }
  }
  static public class Inprogress extends ModelBoxException {
    public Inprogress() {
      super();
    }

    public Inprogress(String message) {
      super(message);
    }
  }
  static public class Already extends ModelBoxException {
    public Already() {
      super();
    }

    public Already(String message) {
      super(message);
    }
  }
  static public class Timedout extends ModelBoxException {
    public Timedout() {
      super();
    }

    public Timedout(String message) {
      super(message);
    }
  }
  static public class Nostream extends ModelBoxException {
    public Nostream() {
      super();
    }

    public Nostream(String message) {
      super(message);
    }
  }
  static public class Reset extends ModelBoxException {
    public Reset() {
      super();
    }

    public Reset(String message) {
      super(message);
    }
  }
  static public class Continue extends ModelBoxException {
    public Continue() {
      super();
    }

    public Continue(String message) {
      super(message);
    }
  }
  static public class Edquot extends ModelBoxException {
    public Edquot() {
      super();
    }

    public Edquot(String message) {
      super(message);
    }
  }
  static public class Stop extends ModelBoxException {
    public Stop() {
      super();
    }

    public Stop(String message) {
      super(message);
    }
  }
  static public class Shutdown extends ModelBoxException {
    public Shutdown() {
      super();
    }

    public Shutdown(String message) {
      super(message);
    }
  }
  static public class Eof extends ModelBoxException {
    public Eof() {
      super();
    }

    public Eof(String message) {
      super(message);
    }
  }
  static public class Noent extends ModelBoxException {
    public Noent() {
      super();
    }

    public Noent(String message) {
      super(message);
    }
  }
  static public class Deadlock extends ModelBoxException {
    public Deadlock() {
      super();
    }

    public Deadlock(String message) {
      super(message);
    }
  }
  static public class Noresponse extends ModelBoxException {
    public Noresponse() {
      super();
    }

    public Noresponse(String message) {
      super(message);
    }
  }
  static public class Io extends ModelBoxException {
    public Io() {
      super();
    }

    public Io(String message) {
      super(message);
    }
  }
}
