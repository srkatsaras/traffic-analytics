import json
import logging
import time

from sqlalchemy import update
from sqlmodel import Session, select

from api.main import Job, ROOT, engine
from scripts.pipeline import process_video


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_job(job_id, **changes):
    with Session(engine) as session:
        job = session.get(Job, job_id)

        if job is None:
            raise RuntimeError(f"Job disappeared: {job_id}")

        for name, value in changes.items():
            setattr(job, name, value)

        session.add(job)
        session.commit()


def claim_job():
    with Session(engine) as session:
        job_id = session.exec(
            select(Job.id)
            .where(Job.status == "queued")
            .limit(1)
        ).first()

        if job_id is None:
            return None

        # Change the status only if the job is still queued.
        result = session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == "queued")
            .values(status="running", progress=0.0, error=None)
        )

        if result.rowcount != 1:
            session.rollback()
            return None

        session.commit()

        job = session.get(Job, job_id)

        return {
            "id": job.id,
            "input_path": job.input_path,
            "x_start": job.x_start,
            "x_end": job.x_end,
            "line_y": job.line_y,
        }


def run_job(job):
    job_id = job["id"]
    output_dir = ROOT / "outputs/jobs" / job_id

    logger.info("Processing job %s", job_id)

    try:
        summary = process_video(
            input_path=job["input_path"],
            output_dir=output_dir,
            x_start=job["x_start"],
            x_end=job["x_end"],
            line_y=job["line_y"],
            on_progress=lambda value: update_job(
                job_id, progress=value
            ),
        )

        # Publish the summary only after processing finishes.
        temporary_path = output_dir / "summary.tmp"
        summary_path = output_dir / "summary.json"

        temporary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(summary_path)

        update_job(
            job_id,
            status="succeeded",
            progress=1.0,
            error=None,
        )
        logger.info("Completed job %s", job_id)

    except KeyboardInterrupt:
        update_job(
            job_id,
            status="failed",
            error="Worker stopped during processing.",
        )
        raise

    except Exception:
        logger.exception("Job %s failed", job_id)
        update_job(
            job_id,
            status="failed",
            error="Processing failed. Check worker logs.",
        )


def main():
    logger.info("Worker ready; waiting for queued jobs.")

    while True:
        job = claim_job()

        if job is None:
            time.sleep(1)
            continue

        run_job(job)


if __name__ == "__main__":
    main()