from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import uuid4
import json

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from sqlmodel import Field, Session, SQLModel, create_engine

from typing import Literal
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "data/api"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MiB

class JobPublic(SQLModel):
    id: str
    status: str = "queued"
    progress: float = 0.0

    x_start: float
    x_end: float
    line_y: float

    width: int
    height: int
    fps: float
    error: str | None = None

class Job(JobPublic, table=True):
    id: str = Field(primary_key=True)
    input_path: str

engine = create_engine(
    f"sqlite:///{STORAGE / 'jobs.db'}",
    connect_args={"check_same_thread": False},
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    STORAGE.mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(engine)
    yield

app = FastAPI(
    title="Traffic Analytics API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/jobs", response_model=JobPublic, status_code=201)
def create_job(
    video: Annotated[UploadFile, File()],
    x_start: Annotated[float, Form(ge=0, le=1)] = 0.10,
    x_end: Annotated[float, Form(ge=0, le=1)] = 0.90,
    line_y: Annotated[float, Form(ge=0, le=1)] = 0.60,
):
    if x_start >= x_end:
        raise HTTPException(422, "x_start must be smaller than x_end.")

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(415, "Unsupported video extension.")

    job_id = str(uuid4())
    input_path = STORAGE / f"{job_id}{suffix}"

    try:
        size = 0
        with input_path.open("xb") as destination:
            while chunk := video.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds 100 MiB.")
                destination.write(chunk)

        if size == 0:
            raise HTTPException(422, "The uploaded file is empty.")

        cap = cv2.VideoCapture(str(input_path))
        try:
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()

        if not ok or fps <= 0:
            raise HTTPException(422, "Could not decode a usable video.")

        height, width = frame.shape[:2]

        job = Job(
            id=job_id,
            input_path=str(input_path),
            x_start=x_start,
            x_end=x_end,
            line_y=line_y,
            width=width,
            height=height,
            fps=fps,
        )

        with Session(engine) as session:
            session.add(job)
            session.commit()
            session.refresh(job)
            response = JobPublic.model_validate(job)

        return response

    except Exception:
        input_path.unlink(missing_ok=True)
        raise

    finally:
        video.file.close()


@app.get("/jobs/{job_id}", response_model=JobPublic)
def get_job(job_id: str):
    with Session(engine) as session:
        job = session.get(Job, job_id)

        if job is None:
            raise HTTPException(404, "Job not found.")

        return JobPublic.model_validate(job)

@app.get("/jobs/{job_id}/results")
def get_results(job_id: str):
    with Session(engine) as session:
        job = session.get(Job, job_id)

        if job is None:
            raise HTTPException(404, "Job not found.")

        if job.status != "succeeded":
            raise HTTPException(
                409,
                f"Results are unavailable; job status is {job.status}.",
            )

        summary_path = (
            ROOT / "outputs/jobs" / job.id / "summary.json"
        )

    if not summary_path.is_file():
        raise HTTPException(500, "Job results are missing.")

    return json.loads(summary_path.read_text(encoding="utf-8"))

@app.get("/jobs/{job_id}/artifacts/{kind}")
def download_artifact(
    job_id: str,
    kind: Literal["video", "detections", "crossings"],
):
    filenames = {
        "video": "annotated.mp4",
        "detections": "detections.csv",
        "crossings": "crossings.csv",
    }

    with Session(engine) as session:
        job = session.get(Job, job_id)

        if job is None:
            raise HTTPException(404, "Job not found.")

        if job.status != "succeeded":
            raise HTTPException(409, "Job has not completed successfully.")

        path = ROOT / "outputs/jobs" / job.id / filenames[kind]

    if not path.is_file():
        raise HTTPException(404, "Artifact not found.")

    return FileResponse(
        path,
        filename=f"{job_id}_{filenames[kind]}",
    )