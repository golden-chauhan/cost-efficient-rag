const DEFAULT_URL = "http://127.0.0.1:8000";

export async function healthCheck(baseUrl = DEFAULT_URL) {
  const url = baseUrl.replace(/\/$/, "");

  const res = await fetch(`${url}/health`);

  if (!res.ok) {
    throw new Error("Backend is not healthy");
  }

  return res.json();
}


export async function askRag(
  question,
  topK = 5,
  baseUrl = DEFAULT_URL
) {
  const start = performance.now();

  const url = baseUrl.replace(/\/$/, "");

  const res = await fetch(`${url}/query`, {
    method: "POST",

    headers: {
      "Content-Type": "application/json",
    },

    body: JSON.stringify({
      question: question,
      top_k: Number(topK),
    }),
  });

  if (!res.ok) {
    let message = `Request failed (${res.status})`;

    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      // Keep default error message
    }

    throw new Error(message);
  }

  const data = await res.json();

  return {
    ...data,
    clientLatency: performance.now() - start,
  };
}