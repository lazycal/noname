
/**
 * The logging utilities.
 */

#pragma once

#include <stdint.h>

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

inline bool ptr_aligned_to(void* ptr, uint64_t size) {
    return ((uint64_t)ptr) % size == 0;
}

#include <iostream>
#include <sstream>

/**
 * Different logging level
 *
 * \def FATAL
 *   Used for fatal and probably irrecoverable conditions
 * \def ERROR
 *   Used for errors which are recoverable within the scope of the function
 * \def WARNING
 *   Logs interesting conditions which are probably not fatal
 * \def EMPH
 *   Outputs as INFO, but in WARNING colors. Useful for
 *   outputting information you want to emphasize.
 * \def INFO
 *   Used for providing general useful information
 * \def DEBUG
 *   Debugging purposes only
 * \def EVERYTHING
 *   Log everything
 */

namespace MYLog {
enum loglevel { NONE = 7, FATAL = 6, ERROR = 5, WARNING = 4, EMPH = 3, INFO = 2, DEBUG = 1, EVERYTHING = 0 };

#ifndef MYLOG_LEVEL
#define MYLOG_LEVEL MYLog::EVERYTHING
#endif

/**
 * Used to determine whether file and line number should be presented
 */
#define MYLOG_FILE_LINE

// logging macro definiations
#define MYLOG(n)          \
    if (n >= MYLOG_LEVEL) \
    MYLog::MessageLogger((char *)__FILE__, __LINE__, n).stream()

// log with tag
#define TMYLOG(n, t)      \
    if (n >= MYLOG_LEVEL) \
    MYLog::MessageLogger((char *)__FILE__, __LINE__, n).stream() << "[" << (t) << "]"

#define MYLOG_IF(n, condition)           \
    if (n >= MYLOG_LEVEL && (condition)) \
    MYLog::MessageLogger((char *)__FILE__, __LINE__, n).stream()

#define ASSERT(condition)       \
    if (unlikely(!(condition))) \
    MYLog::MessageLogger((char *)__FILE__, __LINE__, MYLog::FATAL + 1).stream() << "Assertion! "

#define VERIFY(n, condition) MYLOG_IF(n, (!(condition)))

class MessageLogger {
   public:
    MessageLogger(const char *file, int line, int level) : level_(level) {
        if (level_ < MYLOG_LEVEL)
            return;
        if (level > FATAL) {
            stream_ << "[" << StripBasename(std::string(file)) << ":" << line << "] ";
        } else {
#ifdef MYLOG_FILE_LINE
            stream_ << "[" << StripBasename(std::string(file)) << ":" << line << "] ";
#endif
        }
    }

    ~MessageLogger() {
        if (level_ >= MYLOG_LEVEL) {
            stream_ << "\n";
            // std::cout << "\033[" << DEBUG_LEVEL_COLOR[std::min(level_, 6)] << "m" << stream_.str() << EndcolorFlag();
            std::cout << stream_.str();
            if (level_ >= FATAL)
                abort();
        }
    }

    // Return the stream associated with the logger object.
    std::stringstream &stream() {
        return stream_;
    }

   private:
    std::stringstream stream_;
    int level_;

    // control flags for color
    enum {
        R_BLACK = 39,
        R_RED = 31,
        R_GREEN = 32,
        R_YELLOW = 33,
        R_BULE = 34,
        R_MAGENTA = 35,
        R_CYAN = 36,
        R_WHITE = 37
    };

    const int DEBUG_LEVEL_COLOR[7] = {R_BLACK, R_YELLOW, R_BLACK, R_GREEN, R_MAGENTA, R_RED, R_RED};

    static std::string StripBasename(const std::string &full_path) {
        const char kSeparator = '/';
        size_t pos = full_path.rfind(kSeparator);
        if (pos != std::string::npos) {
            return full_path.substr(pos + 1, std::string::npos);
        } else {
            return full_path;
        }
    }

    static std::string EndcolorFlag() {
        char flag[7];
        snprintf(flag, 7, "%c[0m", 0x1B);
        return std::string(flag);
    }
};

// #define VAR3(begin, n, end) begin << #n << ":" << n << end
#define VAR2(n, end) #n << ":" << n << end
#define VAR(n) #n << ":" << n
}