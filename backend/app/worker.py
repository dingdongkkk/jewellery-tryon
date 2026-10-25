"""Worker process: polls the SQLite queue and runs one generation at a time.

Run: python -m app.worker   (from backend/)
"""
from __future__ import annotations

import logging
import re
import signal
import time
import traceback

from . import config
from .db import db, dumps, init_db, now, row_to_dict
from .providers import ProviderError, QuotaExceededError, TransientProviderError
from .tryon import run_tryon

log = logging.getLogger("worker")
RUNNING_STATES = ("preparing", "generating", "compositing")
_stop = False


def _handle_stop(*_):
    global _stop
    _stop = True


def set_job(jid, **fields):
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    with db() as c:
        c.execute(f"UPDATE tryon_jobs SET {cols} WHERE id=?", (*fields.values(), jid))


def claim_next():
    """Atomically move the oldest queued job to 'preparing'."""
    with db() as c:
        row = c.execute(
            "UPDATE tryon_jobs SET status='preparing', attempts=attempts+1, updated_at=? "
            "WHERE id = (SELECT id FROM tryon_jobs WHERE status='queued' ORDER BY created_at LIMIT 1) "
            "AND status='queued' RETURNING *",
            (now(),),
        ).fetchone()
        return row_to_dict(row)


def quota_message(err: str) -> str:
    if "AI service" in err:
        return err
    wait = re.search(r"(?:try again|retry) in ([0-9:]+(?:\s*\w+)?)", err, re.I)
    msg = (
        "The free AI service's daily limit is used up, so the AI try-on can't run right now."
        + (f" Try again in {wait.group(1)}." if wait else " It resets daily.")
        + " You can still run the try-on without AI lighting (instant)"
        + (", or upgrade the AI service plan for more daily usage." if config.HF_TOKEN
           else ", or add a free HF_TOKEN in backend/.env for more quota.")
    )
    return msg


def process(job: dict):
    jid = job["id"]
    with db() as c:
        product = row_to_dict(c.execute("SELECT * FROM products WHERE id=?", (job["product_id"],)).fetchone())
        upload = row_to_dict(c.execute("SELECT * FROM uploads WHERE id=?", (job["upload_id"],)).fetchone())
    if not product or not upload:
        set_job(jid, status="failed", error="Upload or product no longer exists.")
        return

    attempt = 0
    t0 = time.time()
    while True:
        try:
            result = run_tryon(job, product, upload, config.OUTPUTS_DIR / jid, lambda s: set_job(jid, status=s))
            timings = {**result["timings"], "total": round(time.time() - t0, 2), "info": result["info"]}
            set_job(jid, status="complete", output_path=result["output_path"], seed=result["seed"], timings=dumps(timings), error=None)
            log.info("job %s complete in %.1fs %s", jid, time.time() - t0, result["timings"])
            return
        except QuotaExceededError as e:
            log.warning("job %s quota exceeded: %s", jid, e)
            set_job(jid, status="quota_exceeded", error=quota_message(str(e)))
            return
        except TransientProviderError as e:
            attempt += 1
            if attempt > config.MAX_TRANSIENT_RETRIES:
                set_job(jid, status="failed", error=f"The AI service connection kept failing after {attempt} tries.")
                return
            wait = 5 * 2 ** (attempt - 1)
            log.warning("job %s transient error (attempt %d), retrying in %ss: %s", jid, attempt, wait, e)
            set_job(jid, status="generating", error=f"Connection problem, retrying ({attempt}/{config.MAX_TRANSIENT_RETRIES})...", attempts=job["attempts"] + attempt)
            time.sleep(wait)
        except (ProviderError, ValueError) as e:
            log.warning("job %s failed: %s", jid, e)
            set_job(jid, status="failed", error=str(e)[:800])
            return
        except Exception as e:  # bug: keep the worker alive
            log.error("job %s crashed\n%s", jid, traceback.format_exc())
            set_job(jid, status="failed", error=f"Internal error: {type(e).__name__}: {e}"[:800])
            return


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    init_db()
    with db() as c:  # jobs interrupted by a previous worker crash go back to the queue
        n = c.execute(
            f"UPDATE tryon_jobs SET status='queued' WHERE status IN ({','.join('?' * len(RUNNING_STATES))})", RUNNING_STATES
        ).rowcount
    log.info("worker started (provider default=%s, hf_token=%s, requeued=%d)", config.DEFAULT_PROVIDER, bool(config.HF_TOKEN), n)
    while not _stop:
        job = claim_next()
        if job is None:
            time.sleep(config.POLL_INTERVAL_S)
            continue
        log.info("job %s claimed (%s, product=%s)", job["id"], job["provider"], job["product_id"])
        process(job)


if __name__ == "__main__":
    main()
