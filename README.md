# Traffic Analytics

A video analytics application that detects and tracks vehicles, counts crossings of a configurable horizontal line, and presents results through a web interface.

## Features

- Vehicle detection with YOLOv8 and PyTorch, with ByteTrack tracking.
- Directional counting with a dead band to reduce position jitter.
- Video preview, counting-line placement, and job-status polling.
- Annotated video and downloadable detection and crossing-event CSVs.
- FastAPI backend, a separate processing worker, and persistent job records.
- Docker Compose setup and a GitHub Actions workflow for tests and build checks.

## Tech stack

**Backend:** Python, FastAPI, SQLModel, SQLite, OpenCV, PyTorch, Ultralytics.

**Frontend:** React, TypeScript, Vite.

**Tooling:** Docker, Docker Compose, pytest, GitHub Actions.

## Run locally

Install Docker Desktop and start it. From the project root, run:

```bash
docker compose up --build -d --wait
```

- Application: [http://localhost:8080](http://localhost:8080)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

Upload a stationary-camera traffic video, position the line, and start analysis. Once processing finishes, review the up/down counts and download the results.


To stop the application:

```bash
docker compose down
```

## Tests

Run the Python tests in Docker:

```bash
docker build --target test -t traffic-analytics-tests .
docker run --rm traffic-analytics-tests
```

The CI workflow checks for Python tests, service startup, and frontend-to-API routing.

## Sample footage

Initial development uses Roboflow's [vehicle demo footage](https://media.roboflow.com/supervision/video-examples/vehicles.mp4), listed in the [Supervision assets documentation](https://supervision.roboflow.com/latest/assets/).