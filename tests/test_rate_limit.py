from credit_simulator.rate_limit import RateLimiter


def test_rate_limiter_blocks_until_window_expires():
    limiter = RateLimiter(window_seconds=60)
    assert limiter.check("client", 1, now=10) == (True, 0)
    allowed, retry_after = limiter.check("client", 1, now=20)
    assert not allowed
    assert 49 <= retry_after <= 50
    assert limiter.check("client", 1, now=70) == (True, 0)


def test_zero_limit_is_disabled():
    limiter = RateLimiter()
    assert limiter.check("client", 0, now=10) == (True, 0)


def test_rate_limiter_reports_remaining_capacity():
    limiter = RateLimiter(window_seconds=60)
    assert limiter.check_with_remaining("client", 2, now=10) == (True, 0, 1)
    assert limiter.check_with_remaining("client", 2, now=11) == (True, 0, 0)
    allowed, retry_after, remaining = limiter.check_with_remaining("client", 2, now=12)
    assert not allowed
    assert retry_after > 0
    assert remaining == 0
