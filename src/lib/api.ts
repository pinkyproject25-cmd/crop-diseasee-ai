import type { AnalysisReport, ApiErrorPayload, Language } from "../types";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "");

export class ApiError extends Error {
  code?: string;
  status?: number;

  constructor(message: string, options?: { code?: string; status?: number }) {
    super(message);
    this.name = "ApiError";
    this.code = options?.code;
    this.status = options?.status;
  }
}

function requireApi(): string {
  if (!API_BASE_URL) {
    throw new ApiError(
      "The real AI analysis service has not been configured yet. No prediction was generated.",
      { code: "service_not_configured" },
    );
  }
  return API_BASE_URL;
}

async function readError(response: Response): Promise<ApiError> {
  let payload: ApiErrorPayload | undefined;
  try {
    payload = (await response.json()) as ApiErrorPayload;
  } catch {
    payload = undefined;
  }
  return new ApiError(payload?.detail || "The service could not complete this request.", {
    code: payload?.code,
    status: response.status,
  });
}

export async function analyzeImage(file: File, latitude?: number, longitude?: number): Promise<AnalysisReport> {
  const base = requireApi();
  const form = new FormData();
  form.append("image", file);
  if (latitude !== undefined) form.append("latitude", String(latitude));
  if (longitude !== undefined) form.append("longitude", String(longitude));

  const response = await fetch(`${base}/api/v1/analyze`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(45_000),
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as AnalysisReport;
}

export async function translateReport(report: AnalysisReport, language: Language): Promise<AnalysisReport> {
  if (language === "en") return { ...report, language };
  const base = requireApi();
  const response = await fetch(`${base}/api/v1/reports/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ report, language }),
    signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as AnalysisReport;
}

export async function getReportAudio(report: AnalysisReport, language: Language): Promise<Blob> {
  const base = requireApi();
  const response = await fetch(`${base}/api/v1/reports/speech`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ report, language }),
    signal: AbortSignal.timeout(45_000),
  });
  if (!response.ok) throw await readError(response);
  return response.blob();
}
