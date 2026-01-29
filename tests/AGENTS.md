# tests/ — Pytest Suite

## OVERVIEW
Pytest-based tests for the `gearax` package.

## WHERE TO LOOK
| Area | File | Notes |
|------|------|-------|
| `ConfModule` behavior | `test_modules.py` | Validates abstract base usage (cannot instantiate `ConfModule`) |
| `SubclassRegistryMixin` behavior | `test_mixin.py` | Validates registry lookup + isolation |

## CONVENTIONS
- Function-style tests only (`def test_*():`); no test classes.
- No shared fixtures today (no `conftest.py`); helper classes are defined inline in tests.
- Prefer `pytest.raises(...)` for failure paths; use `match=` when asserting error messages.

## ADDING NEW TESTS
- Name files `test_<module>.py` and keep them small and focused.
- When testing JAX randomness, construct deterministic keys (e.g., `jax.random.key(0)` / `PRNGKey(0)`).
