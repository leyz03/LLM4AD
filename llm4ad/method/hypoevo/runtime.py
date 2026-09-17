"""Call accounting shared by HypoEvo and baseline experiments."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from ...base import LLM


class BudgetExhausted(KeyboardInterrupt):
    """Also exits existing LLM4AD methods that swallow ordinary exceptions."""


class BudgetedLLM(LLM):
    def __init__(self, delegate, max_calls, trace_path, max_total_tokens=None):
        super().__init__(do_auto_trim=False)
        if max_calls < 1:
            raise ValueError('max_calls must be positive')
        self.delegate, self.max_calls = delegate, max_calls
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_total_tokens = max_total_tokens
        self.calls = 0
        self.failures = 0
        self.usage = {}
        self.stop_reason = None
        self.lock = threading.Lock()

    @property
    def remaining(self):
        if self.stop_reason or (self.max_total_tokens is not None and
                               self.usage.get('total_tokens', 0) >= self.max_total_tokens):
            return 0
        return self.max_calls - self.calls

    def draw_sample(self, prompt='', *args, **kwargs):
        with self.lock:
            if self.remaining <= 0:
                self.stop_reason = self.stop_reason or 'budget_exhausted'
                raise BudgetExhausted(self.stop_reason)
            self.calls += 1
            call_id = self.calls
        record = {'call_id': call_id, 'prompt': prompt, 'messages': kwargs.get('messages')}
        started = time.monotonic()
        before = getattr(self.delegate, 'request_count', 0)
        try:
            response = self.delegate.draw_sample(prompt, *args, **kwargs)
            usage = getattr(self.delegate, 'last_usage', {}) or {}
            with self.lock:
                for k, v in usage.items():
                    if isinstance(v, (int, float)):
                        self.usage[k] = self.usage.get(k, 0) + v
            record.update(status='completed', response=response, usage=usage)
            return response
        except Exception as exc:
            self.failures += 1
            self.stop_reason = 'api_error'
            # Do not persist exception text, which could contain provider secrets.
            record.update(status='failed', error_type=type(exc).__name__)
            raise BudgetExhausted('api_error') from exc
        finally:
            record.update(elapsed_seconds=time.monotonic() - started,
                          http_requests=getattr(self.delegate, 'request_count', 0) - before)
            with self.lock, self.trace_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def close(self):
        self.delegate.close()
