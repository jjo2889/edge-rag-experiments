#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

namespace {

struct ServerOptions {
  std::string host = "0.0.0.0";
  int port = 8080;
  std::string rag_query = "./build/rag_query";
  std::string workdir = ".";
  std::string model_config = "configs/model.yaml";
  std::string retrieval_config = "configs/retrieval.yaml";
  std::string request_log_dir = "/tmp/rag_http_requests";
  int timeout_sec = 600;
};

struct ChildResult {
  int exit_code = -1;
  bool timed_out = false;
  std::string stdout_text;
  std::string stderr_text;
};

std::atomic<uint64_t> g_request_counter{0};
std::mutex g_child_mutex;

void printUsage() {
  std::cout << "Usage: rag_http_server [--host host] [--port port] [--rag_query path] [--workdir path]\n"
            << "                       [--model_config path] [--retrieval_config path]\n"
            << "                       [--request_log_dir path] [--timeout_sec seconds]\n";
}

std::string requireValue(int argc, char** argv, int& i, const std::string& arg) {
  if (i + 1 >= argc) {
    throw std::runtime_error("Missing value for " + arg);
  }
  return argv[++i];
}

ServerOptions parseOptions(int argc, char** argv) {
  ServerOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--host") {
      options.host = requireValue(argc, argv, i, arg);
    } else if (arg == "--port") {
      options.port = std::stoi(requireValue(argc, argv, i, arg));
    } else if (arg == "--rag_query") {
      options.rag_query = requireValue(argc, argv, i, arg);
    } else if (arg == "--workdir") {
      options.workdir = requireValue(argc, argv, i, arg);
    } else if (arg == "--model_config") {
      options.model_config = requireValue(argc, argv, i, arg);
    } else if (arg == "--retrieval_config") {
      options.retrieval_config = requireValue(argc, argv, i, arg);
    } else if (arg == "--request_log_dir") {
      options.request_log_dir = requireValue(argc, argv, i, arg);
    } else if (arg == "--timeout_sec") {
      options.timeout_sec = std::stoi(requireValue(argc, argv, i, arg));
    } else if (arg == "--help" || arg == "-h") {
      printUsage();
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }
  return options;
}

std::string nowTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now - seconds).count();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);

  std::tm tm{};
  localtime_r(&now_time, &tm);

  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);

  char output[64];
  std::snprintf(output, sizeof(output), "%s_%06ld", buffer, static_cast<long>(micros));
  return output;
}

std::string zeroPadCounter(uint64_t value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%06llu", static_cast<unsigned long long>(value));
  return buffer;
}

fs::path makeRequestPath(const fs::path& request_log_dir) {
  const uint64_t counter = ++g_request_counter;
  const std::string filename = "request_" + nowTimestamp() + "_" + std::to_string(static_cast<long long>(getpid())) +
                               "_" + zeroPadCounter(counter) + ".json";
  return request_log_dir / filename;
}

void writeFile(const fs::path& path, const std::string& content) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("Failed to open file for write: " + path.string());
  }
  out << content;
}

void setNonBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags == -1) {
    throw std::runtime_error("fcntl(F_GETFL) failed");
  }
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
    throw std::runtime_error("fcntl(F_SETFL) failed");
  }
}

void appendAvailable(int fd, std::string& output) {
  char buffer[4096];
  while (true) {
    const ssize_t n = read(fd, buffer, sizeof(buffer));
    if (n > 0) {
      output.append(buffer, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) {
      return;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return;
    }
    if (errno == EINTR) {
      continue;
    }
    throw std::runtime_error("read from child pipe failed");
  }
}

ChildResult runChild(const ServerOptions& options, const fs::path& request_path) {
  int stdout_pipe[2];
  int stderr_pipe[2];
  if (pipe(stdout_pipe) == -1) {
    throw std::runtime_error("stdout pipe failed");
  }
  if (pipe(stderr_pipe) == -1) {
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    throw std::runtime_error("stderr pipe failed");
  }

  const pid_t pid = fork();
  if (pid == -1) {
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    throw std::runtime_error("fork failed");
  }

  if (pid == 0) {
    close(stdout_pipe[0]);
    close(stderr_pipe[0]);
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stdout_pipe[1]);
    close(stderr_pipe[1]);

    if (chdir(options.workdir.c_str()) == -1) {
      std::cerr << "chdir failed: " << std::strerror(errno) << '\n';
      _exit(127);
    }

    std::vector<std::string> args = {options.rag_query,
                                     "--model_config",
                                     options.model_config,
                                     "--retrieval_config",
                                     options.retrieval_config,
                                     "--request_json",
                                     request_path.string()};
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& arg : args) {
      argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    execvp(argv[0], argv.data());
    std::cerr << "execvp failed: " << std::strerror(errno) << '\n';
    _exit(127);
  }

  close(stdout_pipe[1]);
  close(stderr_pipe[1]);
  setNonBlocking(stdout_pipe[0]);
  setNonBlocking(stderr_pipe[0]);

  ChildResult result;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(options.timeout_sec);
  int status = 0;
  bool exited = false;

  while (!exited) {
    appendAvailable(stdout_pipe[0], result.stdout_text);
    appendAvailable(stderr_pipe[0], result.stderr_text);

    const pid_t wait_result = waitpid(pid, &status, WNOHANG);
    if (wait_result == pid) {
      exited = true;
      break;
    }
    if (wait_result == -1) {
      close(stdout_pipe[0]);
      close(stderr_pipe[0]);
      throw std::runtime_error("waitpid failed");
    }

    if (std::chrono::steady_clock::now() >= deadline) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      result.timed_out = true;
      exited = true;
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  appendAvailable(stdout_pipe[0], result.stdout_text);
  appendAvailable(stderr_pipe[0], result.stderr_text);
  close(stdout_pipe[0]);
  close(stderr_pipe[0]);

  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_code = 128 + WTERMSIG(status);
  }
  return result;
}

nlohmann::json errorJson(const std::string& message) {
  return nlohmann::json{{"status", "error"}, {"error", {{"message", message}}}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ServerOptions options = parseOptions(argc, argv);
    options.request_log_dir = fs::absolute(fs::path(options.request_log_dir)).lexically_normal().string();
    fs::create_directories(options.request_log_dir);

    httplib::Server server;
    server.set_read_timeout(std::chrono::seconds(options.timeout_sec));
    server.set_write_timeout(std::chrono::seconds(options.timeout_sec));

    server.Get("/healthz", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
    });

    server.Get("/readyz", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(nlohmann::json{{"status", "ready"}}.dump(), "application/json");
    });

    server.Post("/query", [&](const httplib::Request& req, httplib::Response& res) {
      try {
        const fs::path request_path = makeRequestPath(options.request_log_dir);
        writeFile(request_path, req.body);

        ChildResult child;
        {
          std::lock_guard<std::mutex> lock(g_child_mutex);
          child = runChild(options, request_path);
        }

        const fs::path stderr_path = request_path.string() + ".stderr.log";
        writeFile(stderr_path, child.stderr_text);
        if (!child.stderr_text.empty()) {
          std::cerr << child.stderr_text;
        }

        if (child.timed_out) {
          res.status = 504;
          res.set_content(errorJson("rag_query timed out").dump(), "application/json");
          return;
        }
        if (child.stdout_text.empty()) {
          res.status = 500;
          res.set_content(errorJson("rag_query produced no stdout").dump(), "application/json");
          return;
        }

        res.status = 200;
        res.set_content(child.stdout_text, "application/json");
      } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(errorJson(e.what()).dump(), "application/json");
      }
    });

    std::cerr << "[http] listening on " << options.host << ':' << options.port << '\n';
    std::cerr << "[http] request_log_dir=" << options.request_log_dir << '\n';
    if (!server.listen(options.host, options.port)) {
      std::cerr << "[http][error] failed to listen on " << options.host << ':' << options.port << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[http][error] " << e.what() << '\n';
    return 1;
  }
}
