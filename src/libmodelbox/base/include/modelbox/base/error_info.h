#ifndef ERROR_INFO_H_
#define ERROR_INFO_H_
#include <iostream>
/**
 * @brief Job error info
 */
struct ErrorInfo {
  /**
   * @brief Job error code
   */
  std::string error_code_;
  /**
   *  @brief Job error message
   */
  std::string error_msg_;
};

#endif
