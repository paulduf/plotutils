# Altair and Polars Integration: Onboarding Notes

## Key Points

- **Altair 5+ supports Polars DataFrames natively**
  - You can pass a `polars.DataFrame` directly to `alt.Chart()` without converting to pandas.
  Example:

```python
import altair as alt
import polars as pl
df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
chart = alt.Chart(df).mark_line()
```

- **No need for pandas conversion**
  - Previous versions required `df.to_pandas()`, but this is no longer necessary.
  - This reduces dependencies and improves performance for Polars users.
- **Testing with Altair and Polars**
  - When writing tests for Altair charts, use Polars DataFrames directly.
  - Ensure your environment has compatible versions of Altair and Polars.

## Additional Notes

- If you encounter errors about data types, check that your Altair version is recent (5.0+).
- For more details, see the [Altair documentation](https://altair-viz.github.io/).

---
This file is intended for onboarding AI agents and developers working with Altair and Polars in this codebase.
