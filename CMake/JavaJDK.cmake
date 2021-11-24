if(${CMAKE_VERSION} VERSION_LESS "3.16.0") 
    find_library(JAVA_AWT_LIBRARY NAMES
        jawt
        HINTS $ENV{JAVA_HOME}
            $ENV{JAVA_HOME}/lib
            $ENV{JAVA_HOME}/lib/amd64
            $ENV{JAVA_HOME}/lib/aarch64
            $ENV{JDK_HOME}
            $ENV{JDK_HOME}/lib
            $ENV{JDK_HOME}/lib/amd64
            $ENV{JDK_HOME}/lib/aarch64
    )

    find_library(JAVA_JVM_LIBRARY NAMES
        jvm
        HINTS $ENV{JAVA_HOME}/lib/server
            $ENV{JAVA_HOME}/lib/amd64/server
            $ENV{JAVA_HOME}/lib/aarch64/server
            $ENV{JAVA_HOME}/jre/lib/server
            $ENV{JAVA_HOME}/jre/lib/amd64/server
            $ENV{JAVA_HOME}/jre/lib/aarch64/server
            $ENV{JDK_HOME}/lib/server
            $ENV{JDK_HOME}/lib/amd64/server
            $ENV{JDK_HOME}/lib/aarch64/server
    )

    find_path(JAVA_INCLUDE_PATH NAMES 
        jni.h
        HINTS $ENV{JAVA_HOME}/include
              $ENV{JDK_HOME}/include
    )
endif()