#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>

namespace booking {

class kiosk;

class ticket {
 public:
  ticket(ticket&& other) noexcept;
  ticket& operator=(ticket&& other) noexcept;
  ~ticket();

  // Non-copyable
  ticket(const ticket&)            = delete;
  ticket& operator=(const ticket&) = delete;

  // Check if ticket is valid
  explicit operator bool() const noexcept { return agent_ != nullptr; }

  // Manually release the ticket
  void release();

 private:
  friend class kiosk;
  explicit ticket(kiosk* agent);

  kiosk* agent_;
};

class kiosk {
 public:
  /// \brief Create unbounded booking agent (unlimited tickets)
  static kiosk unbounded();

  /// \brief Create bounded booking agent (limited concurrent tickets)
  static kiosk bounded(size_t max_tickets);

  ~kiosk() = default;

  /// \brief Blocking: acquire a ticket (waits if bounded and at capacity)
  [[nodiscard]] ticket acquire();

  /// \brief Non-blocking: try to acquire a ticket
  [[nodiscard]] std::optional<ticket> try_acquire();

  /// \brief Try to acquire with timeout
  template <typename Rep, typename Period>
  [[nodiscard]] std::optional<ticket> try_acquire_for(
    const std::chrono::duration<Rep, Period>& timeout)
  {
    std::unique_lock lock(mutex_);

    if (max_tickets_ == 0) {
      // Unbounded mode
      ++active_tickets_;
      ++total_issued_;
      return ticket(this);
    }

    // Bounded mode - wait with timeout
    if (cv_.wait_for(lock, timeout, [this] { return active_tickets_ < max_tickets_; })) {
      ++active_tickets_;
      ++total_issued_;
      return ticket(this);
    }

    return std::nullopt;
  }

  // Wait for all tickets to be released/checked out
  void wait_all();

  // Wait for all tickets with timeout
  template <typename Rep, typename Period>
  bool wait_all_for(const std::chrono::duration<Rep, Period>& timeout)
  {
    std::unique_lock lock(mutex_);
    return wait_cv_.wait_for(lock, timeout, [this] { return active_tickets_ == 0; });
  }

  // Query current state
  size_t active_count() const;
  size_t total_issued() const;
  size_t max_capacity() const;
  bool is_bounded() const;

 private:
  friend class ticket;

  explicit kiosk(size_t max_tickets);
  void release_ticket();

  mutable std::mutex mutex_;
  std::condition_variable cv_;       // For acquire waiting
  std::condition_variable wait_cv_;  // For wait_all

  size_t max_tickets_;  // 0 = unbounded
  size_t active_tickets_{0};
  size_t total_issued_{0};
};

}  // namespace booking
