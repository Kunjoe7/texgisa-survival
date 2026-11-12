"""Durable attempt accounting, including failed and interrupted fits."""
import json
import os


class FitBudgetError(RuntimeError):
    pass


class FitBudgetExceeded(FitBudgetError):
    pass


class FitBudget:
    def __init__(self, path, hard_cap, preflight_fits):
        self.path = path
        self.limit = hard_cap - preflight_fits
        self.attempts = 0
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                value = json.loads(line)  # Corrupt/partial accounting fails closed.
                self.attempts += 1
                if value['attempt'] != self.attempts:
                    raise RuntimeError('Fit ledger sequence mismatch')

    def reserve(self, method, seeds):
        if self.attempts >= self.limit:
            raise FitBudgetExceeded('Total fit hard cap reached, including preflight and retries')
        value = dict(attempt=self.attempts + 1, method=method, seeds=list(seeds),context=getattr(self,'context',{}))
        try:
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(value) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as error:
            raise FitBudgetError('Cannot commit fit accounting; stop before training') from error
        self.attempts += 1
