# SELENE service: contract v1

The SELENE service runs computations of the engine `civiqhub-selene` for a trusted
backend, inside a private network. It keeps no durable state: a job and its result live
in memory for a bounded time and are lost when the service restarts. The backend owns
durable tracking, retries and publication (TechADR-061).

The normative contract is [`openapi.yaml`](openapi.yaml). This page explains it, fixes
the compatibility policy, and gives the interfaces that the CI, the image and the
deployments rely on. Everything here uses synthetic data only.

## Versions

Four versions move separately:

| Version | Where it lives | Changes when |
| --- | --- | --- |
| Contract | `info.version` of `openapi.yaml`, `1.0.0`; the path prefix `/v1` is its major | the HTTP contract changes |
| Service | `version` of [`pyproject.toml`](pyproject.toml) (`selene-api`) | the server changes, for example a compatible fix |
| Engine | `civiqhub_selene.__version__`, and the SHA-256 of the wheel the image installs | the scientific package changes |
| Image | the digest of `ghcr.io/civiqhub/civiqhub-selene` | any of the above, or a dependency, changes |

A compatible fix of the server changes the service version and the image digest, not the
engine: it does not need a release of `civiqhub-selene` on PyPI. Each job reports the
service, contract and engine versions that handled it; `GET /v1/service` adds the wheel
hash, the commit and the deployment label. An image cannot know its own digest: the
deployment proof reads it from the running pod and compares it with the published one.

## Compatibility policy of /v1

- **Additive only.** Under `/v1`, a change may add an optional request field, a
  response field, an operation, a job kind or a payload schema version, an error code,
  or relax a request limit. It never removes or renames a field, never changes the
  meaning of a value, a status code or a job state, and never makes an optional request
  field required.
- **Strict requests, tolerant clients.** The server refuses a request field it does not
  know (`422 invalid_request`). A client ignores a response field it does not know, and
  decides on the status code and the `retryable` flag rather than on an error code it
  does not know.
- **Closed states.** A job is `running`, `succeeded` or `failed`; no state is added
  under `/v1`.
- **Payload schemas.** The envelope (submit, follow, read, errors, versions) does not
  depend on the computation. Each computation is a `kind` with integer
  `schema_version`s; a new incompatible payload is a new `schema_version` served next to
  the old one, not a change of the old one.
- **Breaking changes** open `/v2`, served next to `/v1` during a coexistence window
  announced in the changelog. `/v1` is not changed silently.
- **Proof.** The CI runs the contract client and the fixtures of the approved base
  branch against the new server: a new server that breaks a previous consumer fails,
  whatever its own tests say.

## The job envelope

```
POST   /v1/jobs            submit       202 running job | 200 replay | 409 | 422 | 503 busy
GET    /v1/jobs/{job_id}   follow/read  200 job | 404 job_not_found
DELETE /v1/jobs/{job_id}   forget/stop  204 | 404 job_not_found
GET    /v1/service         versions, kinds, limits, capacity
GET    /healthz            liveness, no token
GET    /readyz             readiness, no token
```

- **Fast acceptance.** `POST /v1/jobs` validates the envelope, the payload schema, the
  sizes and the budget, then starts the computation in a separate process and answers
  at once with a `running` job and a `Location` header.
- **One computation at a time.** There is no queue: a new submission while the slot is
  taken gets `503 busy` with `Retry-After`. `GET` keeps answering during a computation.
- **Submission key.** The caller chooses a `submission_key`; the service binds it to the
  content of the first submission that used it, while that job is retained. The same
  key with the same content returns the same job (`200`), even while busy; the same key
  with other content is `409 submission_conflict`. Checks run in this order:
  authentication, media type, size, JSON, kind and schema version, schema, semantics,
  key, capacity.
- **Content fingerprint.** `content_fingerprint` is `sha256:` and the SHA-256 of the
  canonical JSON (sorted keys, no spaces, UTF-8) of the kind, the schema version and the
  payload with every parameter at its effective value, the votes sorted by cell and the
  listed cells sorted. The key and `caller_reference` are not content. The service
  computes it; the caller stores it.
- **Volatile memory.** A finished job is kept until `expires_at` (at most
  `result_ttl_seconds`), or less when `max_retained_jobs` is reached (the oldest finished
  job goes first), or until a restart. After that, `GET` answers `404 job_not_found`
  whatever the cause, and the key is free again. The service does not promise
  exactly-once execution nor a durable result: the backend decides to resubmit, with the
  same key when it still holds the same input.
- **Caller reference.** `caller_reference` is an opaque text echoed in the job, for
  example the backend job and generation. The service never interprets it: generation
  control stays in the backend.
- **Outcome.** `succeeded` carries `result`, which is either a value or an explained
  absence (`available: false` with `no_votes`, `not_enough_data` or
  `unsupported_input`): an absence is a nominal answer, not an error. `failed` carries
  `error` with `timeout`, `worker_crashed`, `interrupted`, `result_too_large` or
  `engine_error`, and `retryable`.
- **Errors.** Every error body is `{"error": {"code", "message", "retryable",
  "fields"?}}`. Messages and `fields` never echo a value of the request.

## Kind `coclustering.single`, schema 1

A pseudonymised snapshot of the votes on the current revisions of the items of one
project, and the settings of a latent block model (`LatentBlockModel` of the engine).

- **Identifiers.** `voters` are pseudonyms and `items` are `{id, revision}`, both opaque
  and limited to `[A-Za-z0-9._:-]`: no name, no e-mail address, no database key. Their
  order is the order of the rows and columns, and of every array of the result.
- **Scores.** `votes` holds `[row, column, score]` with a product score from -2 to +2.
  The service maps it explicitly to the engine rank; a client never sends a rank.
- **Three cell states.** A cell is *voted* (in `votes`; a score of 0 is a neutral
  vote), *shown without a vote*, or *not shown* (no information). `exposure` says how
  shown cells are known:
  - `complete`: every item was shown to every voter, so a cell without a vote was shown
    without a vote;
  - `listed`: the voted cells and the cells of `shown_without_vote` were shown, every
    other cell was not;
  - `unknown`: the caller does not know. Schema 1 refuses it (`422
    unsupported_exposure`) rather than invent an exposure. The service never reads the
    absence of a vote as an exposure.
- **Settings and seed.** `parameters` are bounded and default to the engine defaults
  (the contract tests keep them equal); `seed` is required. The same snapshot,
  settings, seed and engine wheel give the same result.
- **Result.** `selene.coclustering.single.result/1`: the provenance of the engine
  (method and version, engine version, effective settings, input fingerprint, seed,
  computation fingerprint) and the value (groups, blocks, memberships, co-cluster
  table, objective, integrated likelihood, history). It holds individual memberships:
  it is for the backend only and is never a view to publish as it is. No uncertainty is
  defined for this method, and none is invented.

## Fixtures and contract client

[`contract/fixtures/v1`](contract/fixtures/v1) holds TEST ONLY vectors:

- `coclustering-single-1/reference`: twelve invented voters and six items, two planted
  groups and blocks, neutral votes, cells shown without a vote and cells not shown. Its
  `.expected.json` is the result computed locally with the engine.
- `coclustering-single-1/no-votes`: an explained absence.
- `coclustering-single-1/invalid/*`: each refusal the contract requires, with its status
  and code, and whether the JSON schema alone refuses it.
- `examples/*`: a job in each state and error bodies.

[`contract/make_fixtures.py`](contract/make_fixtures.py) writes them from the engine and
`openapi.yaml`, without the service; `--check` fails when they are stale. A change of the
engine numbers that moves a reference is a scientific change.

[`contract/selene_contract.py`](contract/selene_contract.py) is the consumer side of the
contract, standard library only. It needs a CA bundle for HTTPS and a token file:

```bash
python services/api/contract/selene_contract.py run --url https://HOST:8443 --ca-file ca.pem --token-file token
python services/api/contract/selene_contract.py check --url https://HOST:8443 --ca-file ca.pem --token-file token
```

`run` submits the reference, follows it, compares it with the local reference and prints
a summary without votes, pseudonyms or memberships; `check` runs the consumer suite of
v1 and exits 0 only if every check passes. Both are meant for the synthetic acceptance
test of a deployment as well.

Tests of the contract:

```bash
pip install -e ".[blocks]" --group services/api/pyproject.toml:test
pytest services/api/tests
python services/api/contract/make_fixtures.py --check
```

## Interfaces for the image, the CI and the deployments

The values below are the interface the next tickets implement (SWK-2 to SWK-10) and the
deployments consume (SWK-11 to SWK-17). Limits are technical values to measure on the
real image; they are not privacy thresholds nor a scientific calibration.

**Process.** One server process (no reload, no HTTP worker multiplication) started by
`python -m selene_api`, configured by environment variables only; a start-up check runs
the computation worker once before `/readyz` answers 200. The server refuses to start
without a token file, and without TLS unless it is bound to the loopback interface.

**Inside the server.** Three replaceable layers, each behind a narrow interface:

- HTTP routes and DTOs (`selene_api.http`, `selene_api.contract`): authentication,
  limits, validation and errors; no engine class.
- The job manager (`selene_api.jobs`): admission of one computation, states, idempotency
  and bounded retention, over a `Runner` that starts, follows and stops one computation
  in a supervised child process (one numerical thread, its own process group, a memory
  limit, no core dump).
- The engine adapters (`selene_api.adapters`), one per `(kind, schema_version)`: they
  validate and normalise a payload, run it in the child process by calling the engine,
  and map the engine result to the contract result. No formula is copied out of the
  engine.

**Ports.**

| Port | Protocol | Content | Reached by |
| --- | --- | --- | --- |
| 8443 | HTTPS | `/v1/*`, `/healthz`, `/readyz` | the backend of the same environment, its acceptance job, the kubelet probes |
| 9464 | HTTP | `/metrics`, Prometheus text | the metrics collector only |

**Probes.** Liveness `GET /healthz` and readiness `GET /readyz`, HTTPS on 8443, no
token. Neither fails because a computation runs: busy is a metric and a field of
`GET /v1/service`, never a reason to restart.

**Configuration.**

| Variable | Default | Meaning |
| --- | --- | --- |
| `SELENE_TOKEN_FILE` | required | file of the machine token; one token per line, so that a rotation can list the old and the new; re-read when it changes |
| `SELENE_TLS_CERT_FILE`, `SELENE_TLS_KEY_FILE` | required off loopback | server certificate and key; re-read when they change |
| `SELENE_HOST` / `SELENE_PORT` / `SELENE_METRICS_PORT` | `0.0.0.0` / `8443` / `9464` | listening addresses |
| `SELENE_ENVIRONMENT` | unset | label reported by `GET /v1/service`, for example `dev` or `staging` |
| `SELENE_JOB_TIMEOUT_SECONDS` | `120` | a computation still running is stopped and fails with `timeout` |
| `SELENE_RESULT_TTL_SECONDS` | `900` | retention of a finished job |
| `SELENE_MAX_RETAINED_JOBS` | `16` | finished jobs kept at most |
| `SELENE_MAX_BODY_BYTES` | `8388608` | largest request body |
| `SELENE_MAX_VOTERS` / `SELENE_MAX_ITEMS` / `SELENE_MAX_CELLS` | `2000` / `500` / `250000` | input size limits |
| `SELENE_MAX_WORK` | to measure | computation budget, see `Limits.max_work` |
| `SELENE_MAX_RESULT_BYTES` | `16777216` | largest result |
| `SELENE_WORKER_MEMORY_BYTES` | to measure | address-space limit of the computation process |
| `SELENE_SHUTDOWN_GRACE_SECONDS` | `10` | on SIGTERM: refuse new jobs, stop the computation within this time, exit |

**Container.** Non-root user `10001`, read-only root file system, every capability
dropped, no privilege escalation, no Kubernetes token, no persistent volume; a small
`emptyDir` on `/tmp`. `Recreate` strategy, one replica; `terminationGracePeriodSeconds`
above `SELENE_SHUTDOWN_GRACE_SECONDS`. The token and the certificate are mounted files.

**Image metadata.** OCI labels `org.opencontainers.image.source`, `.revision` (commit),
`.version` (service version), `.licenses`, and `io.civiqhub.selene.contract-version`,
`.engine-version`, `.engine-wheel-sha256`. The CI keeps the tested image as an OCI
archive with a metadata file (manifest digest, commit, versions, wheel hash, test
results) that the publication reuses without rebuilding.

**Acceptance job.** The same image holds the contract client:
`python /app/contract/selene_contract.py check --url ... --ca-file ... --token-file ...`
exits 0 only when contract v1 holds against the deployed service, and prints a JSON
report without individual data.
