
#include "log.h"

namespace modelbox {

LoggerJava::LoggerJava() {}
LoggerJava::~LoggerJava() { UnReg(); }

void LoggerJava::Print(LogLevel level, const char *file, int lineno,
                       const char *func, const char *msg) {
  auto jfile = env_->NewStringUTF(file);
  auto jlineno = (jint)lineno;
  auto jfunc = env_->NewStringUTF(func);
  auto jmsg = env_->NewStringUTF(msg);
  env_->CallObjectMethod(logger_, log_mid_, (jlong)level, jfile, jlineno, jfunc,
                         jmsg);
  env_->DeleteLocalRef(jfile);
  env_->DeleteLocalRef(jfunc);
  env_->DeleteLocalRef(jmsg);
}

void LoggerJava::RegJNICaller(JNIEnv *env, jobject logger) {
  env_ = env;
  logger_ = env->NewGlobalRef(logger);
  jclass cls = env->GetObjectClass(logger_);
  jmethodID mid = env->GetMethodID(
      cls, "jniPrintCallback",
      "(JLjava/lang/String;ILjava/lang/String;Ljava/lang/String;)V");
  log_mid_ = mid;

  env->DeleteLocalRef(cls);
}

void LoggerJava::UnReg() {
  if (env_ == nullptr) {
    return;
  }

  env_->DeleteLocalRef(logger_);
  logger_ = nullptr;
  env_ = nullptr;
}

void LoggerJava::SetLogLevel(LogLevel level) { level_ = level; }

LogLevel LoggerJava::GetLogLevel() { return level_; }

LoggerJavaWapper::LoggerJavaWapper() {}

LoggerJavaWapper::~LoggerJavaWapper() { ModelBoxLogger.SetLogger(nullptr); }

void LoggerJavaWapper::RegLogFunc(std::string pylog) {
  ModelBoxLogger.SetLogger(logger_java_);
}

const std::shared_ptr<Logger> LoggerJavaWapper::GetLogger() {
  return ModelBoxLogger.GetLogger();
}

void LoggerJavaWapper::SetLogger(std::shared_ptr<Logger> logger) {
  ModelBoxLogger.SetLogger(logger);
}

void LoggerJavaWapper::SetLogLevel(LogLevel level) {
  logger_java_->SetLogLevel(level);
}

void LoggerJavaWapper::PrintExt(LogLevel level, const char *file, int lineno,
                                const char *func, const char *msg) {
  ModelBoxLogger.Print(level, file, lineno, func, "%s", msg);
}

}  // namespace modelbox
