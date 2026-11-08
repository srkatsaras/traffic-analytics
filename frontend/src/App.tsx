import './App.css'

import { useEffect, useState } from "react";
import type { FormEvent } from "react";

type Job = {
  id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  progress: number;
  error: string | null;
};

type Results = {
  frames_processed: number;
  counts: {
    up: number;
    down: number;
  };
};

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`/api${path}`, options);

  if (!response.ok) {
    throw new Error(
      `Request failed (${response.status}): ${await response.text()}`,
    );
  }

  return response.json() as Promise<T>;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState("");
  const [lineY, setLineY] = useState(0.6);

  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [results, setResults] = useState<Results | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const busy =
    submitting ||
    job?.status === "queued" ||
    job?.status === "running";

  // Create a local preview without uploading the file.
  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }

    const url = URL.createObjectURL(file);
    setPreview(url);

    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Poll only the currently selected job.
  useEffect(() => {
    if (!jobId) return;

    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;

    async function poll() {
      try {
        const current = await request<Job>(`/jobs/${jobId}`, {
          signal: controller.signal,
        });

        if (controller.signal.aborted) return;

        setJob(current);
        setError(null);

        if (current.status === "succeeded") {
          const summary = await request<Results>(
            `/jobs/${jobId}/results`,
            { signal: controller.signal },
          );

          if (!controller.signal.aborted) setResults(summary);
          return;
        }

        if (current.status === "failed") {
          setError(current.error ?? "Video processing failed.");
          return;
        }
      } catch (err) {
        if (controller.signal.aborted) return;

        setError(
          err instanceof Error ? err.message : "Could not check job status.",
        );
      }

      if (!controller.signal.aborted) {
        timer = setTimeout(poll, 2000);
      }
    }

    void poll();

    return () => {
      controller.abort();
      if (timer !== undefined) clearTimeout(timer);
    };
  }, [jobId]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || busy) return;

    setSubmitting(true);
    setError(null);
    setResults(null);
    setJobId(null);
    setJob(null);

    const form = new FormData();
    form.append("video", file);
    form.append("x_start", "0.10");
    form.append("x_end", "0.90");
    form.append("line_y", String(lineY));

    try {
      const created = await request<Job>("/jobs", {
        method: "POST",
        body: form,
      });

      setJob(created);
      setJobId(created.id);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not submit video.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main>
      <h1>Traffic Analytics</h1>
      <p>Count vehicles crossing a checkpoint in each direction.</p>

      <form onSubmit={submit}>
        <fieldset disabled={Boolean(busy)}>
          <label>
            Traffic video
            <input
              type="file"
              accept=".mp4,.mov,.avi,.mkv"
              required
              onChange={(event) => {
                setFile(event.target.files?.[0] ?? null);
                setJobId(null);
                setJob(null);
                setResults(null);
                setError(null);
              }}
            />
          </label>

          {preview && (
            <>
              <div className="preview">
                <video src={preview} controls playsInline />
                <div
                  className="counting-line"
                  style={{ top: `${lineY * 100}%` }}
                />
              </div>

              <label>
                Line height: {Math.round(lineY * 100)}%
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.01"
                  value={lineY}
                  onChange={(event) =>
                    setLineY(Number(event.target.value))
                  }
                />
              </label>

              <p>
                Place the line across the road where vehicles are clearly visible.
                The line spans 10–90% of the image width.
              </p>
            </>
          )}

          <button type="submit" disabled={!file}>
            {submitting ? "Uploading…" : "Start analysis"}
          </button>
        </fieldset>
      </form>

      {error && <p role="alert">{error}</p>}

      {job && (
        <section aria-live="polite">
          <h2>Analysis status</h2>
          <p>
            {job.status} · {Math.round(job.progress * 100)}%
          </p>
          <progress value={job.progress} max={1} />
          <p className="job-id">Job: {job.id}</p>
        </section>
      )}

      {results && jobId && (
        <section>
          <h2>Results</h2>
          <div className="counts">
            <p>Up <strong>{results.counts.up}</strong></p>
            <p>Down <strong>{results.counts.down}</strong></p>
          </div>

          <p>Frames processed: {results.frames_processed}</p>

          <nav aria-label="Download results">
            <a href={`/api/jobs/${jobId}/artifacts/video`}>
              Annotated video
            </a>
            <a href={`/api/jobs/${jobId}/artifacts/crossings`}>
              Crossing events CSV
            </a>
            <a href={`/api/jobs/${jobId}/artifacts/detections`}>
              Tracked observations CSV
            </a>
          </nav>
        </section>
      )}
    </main>
  );
}

export default App
