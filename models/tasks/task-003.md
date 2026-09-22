Add input validation to the /predict endpoint per specs/api.md's Errors
section — return exactly the status codes listed there (400 for a missing
field naming which one, 422 for an out-of-range value, 500 if the model
failed to load at startup).
