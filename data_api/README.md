# Training Data API

This Flask service serves `synthetic_data.csv` for model-training jobs.

## Run

From the project root:

```text
python -m data_api.app
```

The service listens on `http://localhost:5001`.

## Endpoints

### `GET /health`

Returns a service health response:

```json
{"status": "ok"}
```

### `GET /api/v1/training-data`

Returns the complete dataset by default:

```json
{
  "dataset": "synthetic_data.csv",
  "feature_columns": ["ext_int", "sen_int", "think_feel", "judg_per"],
  "target_column": "mbti_type",
  "row_count": 1000,
  "data": [
    {
      "ext_int": 9,
      "sen_int": 50,
      "think_feel": 68,
      "judg_per": 53,
      "mbti_type": "INTP"
    }
  ]
}
```

Optional query parameters:

- `limit`: return a smaller number of rows, for example `?limit=100`.
- `shuffle`: set to `true` to shuffle the rows before applying `limit`.
- `seed`: set a numeric seed for repeatable shuffled batches, for example `?limit=100&shuffle=true&seed=42`.
- `missing_rate`: probability that each row receives one `null` value, from `0` to `1`.
- `invalid_rate`: probability that each row receives one invalid feature value such as `"unknown"`, from `0` to `1`.

For example, an Airflow task can request intentionally dirty input for the cleaning stage:

```text
GET /api/v1/training-data?limit=1000&shuffle=true&seed=42&missing_rate=0.05&invalid_rate=0.03
```

The API returns rows in the JSON `data` array. Missing values are JSON `null`; invalid feature values are strings. The cleaning task should validate the four `feature_columns`, handle or reject missing values, coerce numeric values, and validate the `target_column` before model training.
