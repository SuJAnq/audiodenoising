import { useEffect, useState } from "react";
import { checkHealth } from "../services/audioApi";

/**
 * Small badge showing whether the backend API is reachable.
 */
export function BackendStatus() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const health = await checkHealth();
      if (!cancelled) setOnline(health.status === "ok");
    };
    check();
    const interval = setInterval(check, 30_000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const cls =
    online === null
      ? "backend-status"
      : online
        ? "backend-status backend-status--online"
        : "backend-status backend-status--offline";

  const label =
    online === null ? "Checking…" : online ? "API Online" : "API Offline";

  return (
    <span className={cls}>
      <span className="backend-status__dot" />
      {label}
    </span>
  );
}
