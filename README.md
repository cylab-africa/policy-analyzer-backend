# Policy Analyzer: Backend

The API behind the [Policy Analyzer](https://github.com/cylab-africa/policy-analyzer-frontend), a tool for analysing data protection and digital identity policy across Africa. It answers questions by retrieving relevant passages from a collection of policy documents and generating an answer from them (retrieval-augmented generation). It also serves the country reference data shown on the frontend's maps and Sources page.

Built with FastAPI, LangChain, Qdrant, MongoDB and Redis.

| Repository | Purpose |
| --- | --- |
| [policy-analyzer-backend](https://github.com/cylab-africa/policy-analyzer-backend) (this repo) | Backend API |
| [policy-analyzer-frontend](https://github.com/cylab-africa/policy-analyzer-frontend) | Next.js web app |


## Architecture

```mermaid
flowchart LR
    client["Website or<br/>API client"] -->|"API key"| api["FastAPI backend"]
    api --> stores[("Qdrant · MongoDB · Redis<br/>SQLite · Disk or S3")]
    api --> services["LLM API<br/>Tavily and email (optional)"]
```

| Store | Holds |
| --- | --- |
| **Qdrant** | Document passages and their embeddings, in separate public and private collections. MongoDB Atlas Vector Search is an experimental alternative (`VECTOR_STORE=mongodb`) |
| **MongoDB** | Country reference data, upload and batch job status, and prompt and topic overrides |
| **Redis** | Cached search results and rate-limit counts. The backend keeps working while Redis is down |
| **SQLite** | Chat history, used for follow-up questions and `GET /api/sessions/{session_id}` (`db_data/` in Docker) |
| **Local disk or S3** | Uploaded documents and batch PDF reports |

### How a question is answered

```mermaid
flowchart LR
    q["Question"] --> prepare["Prepare"] --> search["Search documents<br/>(web if needed)"] --> answer["Stream the<br/>LLM answer"]
```

**Prepare** checks that the question is on a supported topic, resolves follow-ups and detects the topic. An off-topic question gets a short explanation instead. Web search needs `TAVILY_API_KEY`, is off for private documents, and runs when the document search finds too little or the question asks for recent information.

**Document ingestion** runs in the background: extract the text (including descriptions of images in PDFs), detect the country and other details, split the text into passages, then embed and store them. **Batch analysis** runs up to 100 questions through the same answering process, produces a PDF report and can send an email notification.

---

## Setup

| Option | Use it to | Runs in Docker |
| --- | --- | --- |
| [Docker](#docker) | Run the complete backend | API, Qdrant, Redis, MongoDB |
| [Local development](#local-development) | Change the code, with auto-reload | Qdrant, Redis, MongoDB |

**Requirements:** Git, [Docker](https://docs.docker.com/get-docker/) with Compose v2, and an API key for OpenAI or an OpenAI-compatible gateway. If you call OpenAI directly with the Docker option, you also need Python 3. Local development needs [uv](https://docs.astral.sh/uv/getting-started/installation/), which installs Python 3.13.

### Configure

Both options start here:

```bash
git clone https://github.com/cylab-africa/policy-analyzer-backend.git
cd policy-analyzer-backend
cp .env.example .env          # PowerShell: Copy-Item .env.example .env
```

Set these in `.env`:

| Variable | Value |
| --- | --- |
| `API_KEY` | A long random string, for example from `openssl rand -hex 32` |
| `OPENAI_API_KEY` | Your OpenAI or gateway key |
| `MONGODB_CONNECTION_STRING` | Docker: `mongodb://mongo:27017`. Local development: keep the default. Hosted MongoDB: its connection string |

Then choose your model provider:

- **OpenAI directly:** run `python scripts/use_openai_models.py` once. It replaces the gateway model names in `config/llm_providers.json`.
- **A gateway:** set `OPENAI_BASE_URL` in `.env`.

### Docker

```bash
docker compose --profile local-db up -d --build
docker compose exec backend python scripts/seed_reference_data.py   # country data for the website
```

With a hosted MongoDB, leave out `--profile local-db`. To use a port other than 8000, set `BACKEND_PORT` in `.env`.

With `USE_S3=false`, uploaded files and batch reports are kept in the `uploads_data` Docker volume, so they survive when the container is recreated.

### Local development

```bash
uv sync
docker compose up -d mongo qdrant redis
uv run python scripts/seed_reference_data.py
uv run uvicorn main:app --reload --port 8000
```

With a hosted MongoDB, leave `mongo` out of the `docker compose` command.

<details>
<summary>Using pip instead of uv</summary>

```bash
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is generated from `uv.lock`. After changing dependencies, regenerate it with:
`uv export --no-dev --no-hashes --no-emit-project --format requirements-txt -o requirements.txt`

</details>

### Verify

```bash
curl http://localhost:8000/api/monitoring/health -H "X-API-Key: YOUR_API_KEY"
curl http://localhost:8000/api/info/policy_laws -H "X-API-Key: YOUR_API_KEY"
```

The document collection starts empty, so [add documents](#adding-documents) before using the chat. The Qdrant dashboard is at <http://localhost:6333/dashboard>. For interactive API docs at <http://localhost:8000/api/docs>, set `ENABLE_API_DOCS=true` in `.env` on your own machine only, never in production.

### Stop

```bash
docker compose --profile local-db down       # stop and keep the data
docker compose --profile local-db down -v    # stop and DELETE the Docker volumes
docker compose logs -f backend               # follow the API logs
```

---

## Adding documents

The repository does not include the policy document collection. Add your own laws, regulations and reports (PDF, DOCX, TXT or MD). Each document is assigned to a topic area: `data-protection`, `digital-id` or `general`.

**Upload files through the API**

```bash
curl -X POST "http://localhost:8000/api/documents/upload-documents?tab=data-protection" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "files=@kenya-data-protection-act-2019.pdf"
```

**Process a whole folder on the server**

```bash
curl -X POST http://localhost:8000/api/documents/folder-process \
  -H "X-API-Key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"folder_path": "/path/to/documents", "tab": "data-protection"}'
```

This route is switched off until `INGEST_FOLDER_ROOT` in `.env` names the one directory it may read from, and `folder_path` must be inside it (a relative path is read relative to it). The path is read by the API process. With Docker it must be a path inside the container, for example a folder you mount into `/app/uploads` and set as `INGEST_FOLDER_ROOT=/app/uploads`.

Processing runs in the background. The model extracts each document's country, title and description, then splits the text into passages and stores them in Qdrant. You can also upload documents from the frontend's Policy Chat page.

---

## Updating the reference data

The maps and the Sources page read the country records from MongoDB, not from the JSON files. The files in `app/data/` (`countries_law_data.json` and `accurate-digital-id-data.json`) are the seed data for a new install and the reviewed copy of those records. To change what the site shows:

1. Edit the JSON files and include a link to a primary source for every change.
2. Preview the effect on a database: `uv run python scripts/seed_reference_data.py --update --dry-run -v` (`-v` lists each differing field).
3. Apply it: `uv run python scripts/seed_reference_data.py --update`

`--update` matches records by `isoAlpha2` and replaces the stored record, so edits made through the `/api/info` endpoints are overwritten. Without `--update` the script only loads empty collections.

If the database holds newer data than the JSON files (for example because records were edited through the API), save its records first with `uv run python scripts/seed_reference_data.py --export exported`, then merge the exported files into `app/data/` before running `--update`.

---

## Connecting the frontend

In the [frontend](https://github.com/cylab-africa/policy-analyzer-frontend)'s `.env`, set:

```bash
BACKEND_API_URL=http://localhost:8000/api
BACKEND_API_KEY=<the same value as API_KEY in this repo's .env>
```

The frontend calls this API from its own server and adds the key there, so the key never reaches visitors' browsers.

---

## Configuration

### Environment variables

See [.env.example](.env.example) for the full list of settings.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `API_KEY` | Yes | | Key clients send in `X-API-Key` (`Authorization: Bearer` also works) |
| `OPENAI_API_KEY` | Yes | | OpenAI key, or your gateway's key |
| `MONGODB_CONNECTION_STRING` | Yes | | MongoDB URI (local or Atlas) |
| `OPENAI_BASE_URL` | No | OpenAI | OpenAI-compatible gateway URL |
| `VECTOR_STORE` | No | `qdrant` | `qdrant`, or `mongodb` for Atlas Vector Search |
| `QDRANT_HOST` | No | `http://localhost:6333` | Qdrant URL |
| `REDIS_HOST` / `REDIS_PORT` | No | `localhost` / `6379` | Query cache and rate-limit counts. Searches still work while Redis is down, just without the cache |
| `USE_S3` | No | `true` in code, `false` in `.env.example` | Store uploads in S3 instead of on disk |
| `TAVILY_API_KEY` | No | | Enables the web-search fallback |
| `SMTP_*`, `SENDER_EMAIL` | No | `SENDER_EMAIL` = `SMTP_USERNAME` | Email delivery of batch reports. Emails are sent from `SMTP_USERNAME` unless `SENDER_EMAIL` names an alias of that account |
| `CORS_ALLOWED_ORIGINS` | No | none | Comma-separated browser origins allowed to call the API directly. Normally empty, because the website calls it from its server. `*` is ignored |
| `ENABLE_API_DOCS` | No | `false` | Serve `/api/docs` and `/openapi.json` (public, no key). For local development only |
| `REDIS_PASSWORD` | No | none | Password for Redis. Docker Compose starts Redis with it, and the backend uses the same value. Set it in production |
| `MONGO_ROOT_USERNAME`, `MONGO_ROOT_PASSWORD` | No | none | Login for the bundled MongoDB (`--profile local-db`). Set both, or neither (no login: local use only). Use letters and digits only, because the password goes into `MONGODB_CONNECTION_STRING` (`mongodb://USER:PASSWORD@mongo:27017/?authSource=admin`). Takes effect only when the data volume is first created |
| `INGEST_FOLDER_ROOT` | No | unset | Directory that `POST /api/documents/folder-process` may read from. The route is switched off while this is unset |
| `ALLOWED_FILE_TYPES` | No | PDF, text, Markdown, Word | Comma-separated MIME types accepted for upload. An upload's type comes from its file extension (`.pdf`, `.txt`, `.md`, `.docx`), because browsers disagree on what they send for some files. The old Word `.doc` format cannot be read and is always refused |
| `QUERY_RATE_LIMIT_PER_MINUTE` | No | `20` | Questions per client per minute on `/api/query_v2` (`0` = no limit) |
| `BATCH_UPLOAD_RATE_LIMIT_PER_HOUR` | No | `10` | Batch uploads per client per hour (`0` = no limit) |
| `UPLOAD_RATE_LIMIT_PER_HOUR` | No | `10` | Document uploads per client per hour (`0` = no limit). One upload holds at most 10 files |
| `RATE_LIMIT_STORE` | No | `redis` | Where rate-limit counts are kept: `redis` (shared by all workers and replicas, in memory while Redis is down) or `memory` (per process) |
| `TRUSTED_PROXY_IPS` | No | none | The website's server as the API sees it: exact addresses or hostnames (such as its container name on a shared Docker network, re-resolved every 30 s so a recreated container is still recognised). Requests from them may name the visitor in `X-Client-IP`, so rate limits apply per visitor. Address ranges are refused, because a Docker network range includes the gateway that Docker forwards some outside connections from. While empty, everyone using the website shares one limit, and the log warns about it |
| `ADMIN_API_KEY` | No | unset | Operator key (`X-Admin-Key` header) for the `/api/admin` routes, listing all batch jobs, deleting any job, and changing the country reference data (`POST`, `PATCH`, `DELETE` under `/api/info`). Use a long random value (`openssl rand -hex 32`). Must differ from `API_KEY`; never give it to the website. While unset (or equal to `API_KEY`), all of these are switched off |

### Configuration files

| File | Purpose |
| --- | --- |
| `config/application.json` | Topic areas ("tabs"), metadata detection and retrieval parameters |
| `config/llm_providers.json` | Providers, and the model used for each task |
| `config/prompts.json` | Prompt templates |

### Adapting to another topic

Topic areas ("tabs") are defined in `config/application.json` and listed in `supported_tabs` in `.env`; the instructions given to the model are in `config/prompts.json` under `system_prompts`. Data protection and digital identity are also named in a few places in the code, so adding or replacing a topic needs small code changes as well:

- the allowed `tab` values in `app/routes/document_processing.py`;
- the document types in `app/schema/pydantic_models.py`, and where they are used for detection (`app/services/entity_detector.py`, `app/chat/llm_chat.py`);
- the wording of the `router` prompt in `config/prompts.json`, which describes what the document collection contains;
- in the frontend, the `ChatTab` values and the topic screens.

New tabs can also be created at runtime with `POST /api/admin/tabs/create`, but the website will not show a tab until the frontend knows about it.

---

## Deployment

### Single container

Build the image and point it at managed MongoDB, Qdrant and Redis instances:

```bash
docker build -t policy-analyzer-backend .
docker run -d --name pdp-backend -p 8000:8000 --env-file .env policy-analyzer-backend
```

### One server with Docker Compose

`docker-compose.yaml` runs the API, Qdrant and Redis, plus MongoDB either hosted elsewhere or bundled:

```bash
docker compose up -d --build                      # MongoDB hosted elsewhere
docker compose --profile local-db up -d --build   # or with the bundled MongoDB
```

With the bundled MongoDB in production, set `MONGO_ROOT_USERNAME` and `MONGO_ROOT_PASSWORD` in `.env` before the first start, and use `mongodb://USER:PASSWORD@mongo:27017/?authSource=admin` as `MONGODB_CONNECTION_STRING`. To start it with every `docker compose up`, add `COMPOSE_PROFILES=local-db` to `.env`. Nothing backs the bundled database up for you, so schedule `scripts/backup_mongo.sh`. It writes a compressed copy to `~/pdp-backups` (change it with `BACKUP_DIR`), keeps 14 days (`KEEP_DAYS`), and exits with an error if the dump is empty. Run it every night with `crontab -e`:

```bash
30 2 * * * /path/to/policy-analyzer-backend/scripts/backup_mongo.sh >> $HOME/pdp-backups/backup.log 2>&1
```

A backup on the same server is lost with the server, so also copy `~/pdp-backups` somewhere else (another machine or a storage bucket). To restore a backup into the running container:

```bash
gunzip -c ~/pdp-backups/mongo_YYYY-MM-DD_HHMMSS.archive.gz | docker exec -i pdp-mongo sh -c 'mongorestore --archive -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin'
```

Put a reverse proxy with HTTPS (such as Nginx or Caddy) in front of the API port, and back up the `qdrant_data` volume (plus `mongo_data` if you use the bundled MongoDB, and `uploads_data` if `USE_S3=false`) and the `db_data/` folder. Redis, Qdrant and MongoDB are published on `127.0.0.1` only; keep it that way.

### Updating a running deployment

```bash
git pull --ff-only
docker compose up -d --build
```

- **Upgrading a deployment that has no `uploads_data` volume, with `USE_S3=false`?** Its files are inside the old container. Copy them out *before* updating with `docker cp pdp-backend:/app/uploads ./uploads-backup`. After updating, copy them back with `docker cp ./uploads-backup/. pdp-backend:/app/uploads/`, then run `docker compose exec -u 0 backend chown -R 10001:10001 /app/uploads`.
- **Back up first:** `.env`, `config/`, `db_data/`, and the Qdrant volume (plus the uploads volume if `USE_S3=false`). `docker volume ls` shows the volume names; copy it with `docker run --rm -v <volume>:/data -v "$PWD":/backup alpine tar czf /backup/qdrant_volume.tgz -C /data .`.
- **Changed `.env`?** Run `docker compose up -d --force-recreate backend`. A plain `docker compose restart` does not reload it.
- **Behind a reverse proxy?** Set `BACKEND_PORT` in `.env` to the port the proxy forwards to (the default is `8000`). If you leave it out, the backend moves to a different port and the proxy returns `502`.
- **Never** run `docker compose down -v` or `docker volume prune`. They can delete vectors, cached data and uploaded files.
- **Recreated a container the proxy reaches by name?** Restart the proxy afterwards, so that it looks the address up again.

### Rotating the API key

The key lives in two places: `API_KEY` here and `BACKEND_API_KEY` in the frontend's environment. Both must change together, so expect a few seconds of `403` errors:

1. Set the new `API_KEY` in `.env` and run `docker compose up -d --force-recreate backend`.
2. Restart the frontend container with the new `BACKEND_API_KEY` straight away. No rebuild is needed, because the frontend reads it when it starts.

### Before exposing an instance publicly

- **The API key is shared, not per user.** Every endpoint is behind the single `API_KEY`. The frontend keeps it on its server, but anyone who can reach the API directly with the key can use every route that needs nothing more, including document processing. Keep the API off the public internet, reachable only from the frontend's server, and block `/api/admin` and `/api/documents` at your reverse proxy. The `/api/admin` routes and the write methods on `/api/info` also need `ADMIN_API_KEY`, and the website's proxy passes no admin requests and only `GET` requests to `/api/info`.
- **Batch jobs have an owner token.** Creating a batch job returns a `delete_token` once; deleting the job needs it in the `X-Job-Token` header. Listing jobs (`GET /api/batch/jobs`) is for the operator only and needs `ADMIN_API_KEY` in the `X-Admin-Key` header; the same key can delete any job, including jobs created before tokens existed.
- **Folder processing is off by default.** `POST /api/documents/folder-process` only works once `INGEST_FOLDER_ROOT` names a directory, and only for folders inside it.
- **Rate limits** apply to questions, document uploads and batch uploads (see the environment variables above). They are counted in Redis over a sliding window, so they hold across workers and restarts (per process while Redis is down), and IPv6 clients are counted per /64. An upload counts only once its files have passed validation, and a document upload holds at most 10 files. Set `TRUSTED_PROXY_IPS` to the website server's address or container name, or all of its visitors share one limit (20 questions a minute by default).
- **CORS** allows no browser origin unless you list one in `CORS_ALLOWED_ORIGINS`.
- **The container runs as a non-root user** (UID 10001). Docker Compose hands `./db_data` to that user before the backend starts (the `db-data-permissions` service). If you run the image without Compose and bind-mount `./db_data`, run `sudo chown -R 10001:10001 db_data` once, or the chat history database cannot be written.
- **Use S3 for uploads** in multi-instance deployments (`USE_S3=true` plus the `AWS_*` variables).

---

## API overview

All routes except `/` require the API key. When `ENABLE_API_DOCS=true`, `/api/docs` and `/openapi.json` are also served without a key and show the full request and response schemas. Leave it off in production.

| Method and path | Purpose |
| --- | --- |
| `POST /api/query_v2` | Ask a question; streams the answer as Server-Sent Events |
| `GET /api/sessions/{session_id}` | Conversation history |
| `POST /api/documents/upload-documents` | Upload documents for processing |
| `POST /api/documents/folder-process` | Process every document in a server folder |
| `POST /api/batch/upload` | Submit an Excel file of up to 100 questions (`visibility=private` needs a `client_id`) |
| `GET /api/batch/status/{job_id}` | Batch job status |
| `GET /api/batch/download/{job_id}` | Download a batch PDF report |
| `GET /api/info/policy_laws`, `GET /api/info/digital_id` | Country reference data (`POST`, `PATCH` and `DELETE` need `X-Admin-Key`) |
| `/api/admin/...` | Prompt, tab and cache management (needs `X-Admin-Key`) |
| `GET /api/batch/jobs`, `DELETE /api/batch/{job_id}` | List all batch jobs (needs `X-Admin-Key`); delete a job (needs its `X-Job-Token`, or `X-Admin-Key`) |
| `GET /api/monitoring/health` | Provider health and circuit-breaker state |

Example query:

```bash
curl -N -X POST http://localhost:8000/api/query_v2 \
  -H "X-API-Key: YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"query": "What are the data subject rights under Kenya'\''s Data Protection Act?", "tab": "data-protection"}'
```

---

## Project structure

```
├── app/
│   ├── core/            # Settings and dependency container
│   ├── data/            # Country reference data (seeded into MongoDB)
│   ├── db/              # MongoDB and chat-history clients
│   ├── middleware/      # API-key authentication
│   ├── routes/          # Admin, batch and document-processing endpoints
│   └── services/        # Query handling, retrieval, document processing, LLM factory
├── config/              # application.json, llm_providers.json, prompts.json
├── scripts/             # Setup helpers (seed data, switch model names)
├── main.py              # FastAPI entry point
├── Dockerfile
└── docker-compose.yaml       # API + Qdrant + Redis (+ optional local MongoDB)
```

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Startup fails with `Failed to connect to MongoDB` | Check MongoDB is running (`docker compose ps`) and `MONGODB_CONNECTION_STRING` is correct |
| `model not found` or 404 errors from OpenAI | Run `python scripts/use_openai_models.py`, or set `OPENAI_BASE_URL` for your gateway |
| Maps or Sources page are empty | Run `scripts/seed_reference_data.py` ([Setup](#setup)) |
| Chat says it has no information | The document collection is empty; see [Adding documents](#adding-documents) |
| Searches are slow or repeated questions are not cached | Redis is not reachable; searches carry on without the query cache (check `REDIS_HOST` and `REDIS_PASSWORD`) |
| `401 API key required` | Send the `X-API-Key` header with the value of `API_KEY` |
| Uploads fail with S3 errors | Set `USE_S3=false` for local storage |
| `502 Bad Gateway` from the reverse proxy after an update | The backend is on a different port than the proxy expects (set `BACKEND_PORT`), or the proxy needs a restart after a container was recreated |
| `403 Invalid API key` after changing `API_KEY` | The frontend is still using the old key. Restart it with the new one ([Rotating the API key](#rotating-the-api-key)) |

---

## License

Released under the [MIT License](LICENSE).

## Contributing

Issues and pull requests are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) first, and note that everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). To report a security problem, follow [SECURITY.md](SECURITY.md) instead of opening a public issue. If you use this software in your work, see [CITATION.cff](CITATION.cff).
