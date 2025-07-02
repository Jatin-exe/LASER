#pragma once
#include <fmt/format.h>
#include <functional>
#include <iostream>


//This allows us to use nice fmt style args with a retargetable log function

namespace laser::tracking::logging {
enum class LogLevel { INFO, WARNING, ERROR, DEBUG, TRACE };
void default_log_function(const std::string &message, LogLevel level);
void set_log_function(
    std::function<void(const std::string &, LogLevel)> log_func);
extern std::function<void(const std::string &, LogLevel)> log_function;
template <typename... Args>
void info(fmt::format_string<Args...> fmt, Args &&...args) {
  log_function(fmt::format(fmt, std::forward<Args>(args)...), LogLevel::INFO);
}
template <typename... Args>
void warn(fmt::format_string<Args...> fmt, Args &&...args) {
  log_function(fmt::format(fmt, std::forward<Args>(args)...),
               LogLevel::WARNING);
}
template <typename... Args>
void error(fmt::format_string<Args...> fmt, Args &&...args) {
  log_function(fmt::format(fmt, std::forward<Args>(args)...), LogLevel::ERROR);
}
template <typename... Args>
void debug(fmt::format_string<Args...> fmt, Args &&...args) {
  log_function(fmt::format(fmt, std::forward<Args>(args)...), LogLevel::DEBUG);
}

template <typename... Args>
void trace(fmt::format_string<Args...> fmt, Args &&...args) {
  log_function(fmt::format(fmt, std::forward<Args>(args)...), LogLevel::TRACE);
}

} // namespace laser::tracking::logging