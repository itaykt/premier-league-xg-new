import { useEffect, useState } from "react";
import type { AppDataPayload } from "../types";

export function useAppData(url = "/app_data.json") {
  const [data, setData] = useState<AppDataPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load ${url}`);
        return r.json();
      })
      .then((j: AppDataPayload) => {
        if (!cancelled) setData(j);
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [url]);

  return { data, error };
}
