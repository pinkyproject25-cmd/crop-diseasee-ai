import type { AnalysisReport } from "../types";

const HISTORY_KEY = "crop-disease-ai.history.v1";
const DRAFT_IMAGE_KEY = "crop-disease-ai.draft-image.v1";
const ACTIVE_REPORT_KEY = "crop-disease-ai.active-report.v1";

export function loadHistory(): AnalysisReport[] {
  try {
    const value = localStorage.getItem(HISTORY_KEY);
    return value ? (JSON.parse(value) as AnalysisReport[]) : [];
  } catch {
    return [];
  }
}

export function saveReport(report: AnalysisReport): void {
  const current = loadHistory().filter((item) => item.id !== report.id);
  localStorage.setItem(HISTORY_KEY, JSON.stringify([report, ...current].slice(0, 30)));
  sessionStorage.setItem(ACTIVE_REPORT_KEY, JSON.stringify(report));
}

export function deleteReport(id: string): void {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(loadHistory().filter((item) => item.id !== id)));
}

export function clearHistory(): void {
  localStorage.removeItem(HISTORY_KEY);
}

export function setActiveReport(report: AnalysisReport): void {
  sessionStorage.setItem(ACTIVE_REPORT_KEY, JSON.stringify(report));
}

export function getActiveReport(): AnalysisReport | null {
  try {
    const value = sessionStorage.getItem(ACTIVE_REPORT_KEY);
    return value ? (JSON.parse(value) as AnalysisReport) : null;
  } catch {
    return null;
  }
}

export function setDraftImage(dataUrl: string): void {
  sessionStorage.setItem(DRAFT_IMAGE_KEY, dataUrl);
}

export function getDraftImage(): string | null {
  return sessionStorage.getItem(DRAFT_IMAGE_KEY);
}

export function clearDraftImage(): void {
  sessionStorage.removeItem(DRAFT_IMAGE_KEY);
}
