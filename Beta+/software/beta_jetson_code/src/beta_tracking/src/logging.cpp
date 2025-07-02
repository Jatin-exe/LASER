#include "beta_tracking/logging.h"

namespace laser::tracking::logging {
std::function<void(const std::string &, LogLevel)> log_function;
void set_log_function(
    std::function<void(const std::string &, LogLevel)> log_func) {
  log_function = std::move(log_func);
}
} // namespace laser::tracking::logging
