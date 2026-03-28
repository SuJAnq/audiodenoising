import { useEffect, useState } from "react";
import { checkHealth } from "../services/audioApi";

/**
 * Small badge showing whether the backend API is reachable.
 */
export function BackendStatus() {
  const [status, setStatus] = useState<"checking" | "online" | "warming" | "offline">("checking");
  const [consecutiveErrors, setConsecutiveErrors] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const health = await checkHealth();

      if (cancelled) return;

      if (health.status === "ok") {
        setConsecutiveErrors(0);
        setStatus("online");
        return;
      }

      if (health.status === "warming") {
        setConsecutiveErrors((prev) => prev + 1);
        setStatus("warming");
        return;
      }

      setConsecutiveErrors((prev) => {
        const next = prev + 1;
        // Avoid flashing Offline during short startup/network blips.
        setStatus(next >= 3 ? "offline" : "warming");
        return next;
      });
    };

    check();
    const interval = setInterval(check, 20_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const cls =
    status === "online"
      ? "backend-status backend-status--online"
      : status === "offline"
        ? "backend-status backend-status--offline"
        : status === "warming"
          ? "backend-status backend-status--warming"
          : "backend-status";

  const label =
    status === "online"
      ? "API Online"
      : status === "offline"
        ? "API Offline"
        : status === "warming"
          ? "Waking API…"
          : "Checking…";

  return (
    <span className={cls}>
      <span className="backend-status__dot" />
      {label}
    </span>
  );
}
