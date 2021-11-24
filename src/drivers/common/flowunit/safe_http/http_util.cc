#include "http_util.h"

uint64_t HttpRequestLimiter::max_request_;
std::atomic_size_t HttpRequestLimiter::request_count_;
std::mutex HttpRequestLimiter::request_mutex_;

void SafeReply(const web::http::http_request &request,
               web::http::status_code status) {
  auto resp = web::http::http_response(status);
  resp.headers().add(U("Referrer-Policy"),
                     U("strict-origin-when-cross-origin"));
  resp.headers().add(
      U("Content-Security-Policy"),
      U("default-src 'self'  data: 'unsafe-inline' 'unsafe-eval'; "
        "objectsrc 'none'; "
        "frame-ancestors 'none'"));
  resp.headers().add(U("X-Frame-Options"), U("DENY"));
  request.reply(resp).then([](pplx::task<void> t) { HandleError(t); });
}

void SafeReply(const web::http::http_request &request,
               web::http::status_code status, const utf8string &body_data) {
  auto resp = web::http::http_response(status);
  resp.set_body(body_data);
  resp.headers().add(U("Referrer-Policy"),
                     U("strict-origin-when-cross-origin"));
  resp.headers().add(
      U("Content-Security-Policy"),
      U("default-src 'self'  data: 'unsafe-inline' 'unsafe-eval'; "
        "objectsrc 'none'; "
        "frame-ancestors 'none'"));
  resp.headers().add(U("X-Frame-Options"), U("DENY"));
  request.reply(resp).then([](pplx::task<void> t) { HandleError(t); });
}

void SafeReply(const web::http::http_request &request,
               web::http::status_code status,
               const concurrency::streams::istream &body_data,
               const utility::string_t &content_type) {
  auto resp = web::http::http_response(status);
  resp.set_body(body_data, content_type);
  resp.headers().add(U("Referrer-Policy"),
                     U("strict-origin-when-cross-origin"));
  resp.headers().add(
      U("Content-Security-Policy"),
      U("default-src 'self'  data: 'unsafe-inline' 'unsafe-eval'; "
        "objectsrc 'none'; "
        "frame-ancestors 'none'"));
  resp.headers().add(U("X-Frame-Options"), U("DENY"));
  request.reply(resp).then([](pplx::task<void> t) { HandleError(t); });
}

utility::string_t GetSupportedMethods() {
  utility::string_t allowed;
  std::vector<web::http::method> methods = {
      web::http::methods::POST, web::http::methods::GET,
      web::http::methods::DEL, web::http::methods::PUT};
  bool first = true;
  for (auto iter = methods.begin(); iter != methods.end(); ++iter) {
    if (!first) {
      allowed += U(", ");
    } else {
      first = false;
    }
    allowed += (*iter);
  }
  return allowed;
}

void HandleError(pplx::task<void> &t) {
  try {
    t.get();
  } catch (const std::exception &e) {
    MBLOG_ERROR << "http error" << e.what();
  }
}

void HandleUnSupportMethod(web::http::http_request request) {
  web::http::http_response response(web::http::status_codes::MethodNotAllowed);
  response.headers().add(U("Allow"), GetSupportedMethods());
  request.reply(response);
}