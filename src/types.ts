export type AnalysisState = "healthy" | "diseased" | "unknown";
export type Severity = "Low" | "Medium" | "High" | null;
export type Language = "en" | "te" | "hi";

export interface Prediction {
  label: string;
  probability: number;
}

export interface WeatherSnapshot {
  location: string;
  observedAt: string;
  temperatureC: number | null;
  humidityPercent: number | null;
  precipitationMm: number | null;
  description: string;
}

export interface AnalysisReport {
  id: string;
  createdAt: string;
  modelVersion: string;
  state: AnalysisState;
  crop: string | null;
  disease: string | null;
  condition: string;
  confidence: number | null;
  diseaseRate: number | null;
  severity: Severity;
  healthScore: number | null;
  observedSymptoms: string[];
  typicalSymptoms: string[];
  causes: string[];
  recommendations: string[];
  uncertaintyReason: string | null;
  topPredictions: Prediction[];
  weather: WeatherSnapshot | null;
  thumbnailDataUrl?: string;
  language?: Language;
}

export interface ApiErrorPayload {
  detail?: string;
  code?: string;
}
