"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useSyncExternalStore,
} from "react";

type ThemeMode = "light" | "dark" | "system";
type ResolvedMode = "light" | "dark";

interface ThemeModeContextValue {
  mode: ThemeMode;
  resolvedMode: ResolvedMode;
  setMode: (mode: ThemeMode) => void;
  mounted: boolean;
}

const STORAGE_KEY = "theme-mode";
const DEFAULT_MODE: ThemeMode = "system";

const ThemeModeContext = createContext<ThemeModeContextValue | null>(null);

function subscribeToMediaQuery(callback: () => void) {
  const mql = window.matchMedia("(prefers-color-scheme: dark)");
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function getSystemSnapshot(): ResolvedMode {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function getServerSnapshot(): ResolvedMode {
  return "light";
}

const noop = () => () => {};

let storageModeListeners: Array<() => void> = [];

function subscribeToStorageMode(callback: () => void) {
  storageModeListeners.push(callback);
  return () => {
    storageModeListeners = storageModeListeners.filter((l) => l !== callback);
  };
}

function notifyStorageModeChange() {
  storageModeListeners.forEach((l) => l());
}

function getStoredModeSnapshot(): ThemeMode {
  return (localStorage.getItem(STORAGE_KEY) as ThemeMode) ?? DEFAULT_MODE;
}

function getStoredModeServerSnapshot(): ThemeMode {
  return DEFAULT_MODE;
}

export function ThemeModeProvider({ children }: { children: React.ReactNode }) {
  const mode = useSyncExternalStore(
    subscribeToStorageMode,
    getStoredModeSnapshot,
    getStoredModeServerSnapshot,
  );

  const mounted = useSyncExternalStore(
    noop,
    () => true,
    () => false,
  );

  const systemPreference = useSyncExternalStore(
    subscribeToMediaQuery,
    getSystemSnapshot,
    getServerSnapshot,
  );

  const resolvedMode: ResolvedMode =
    mode === "system" ? systemPreference : mode;

  const setMode = useCallback((next: ThemeMode) => {
    localStorage.setItem(STORAGE_KEY, next);
    notifyStorageModeChange();
  }, []);

  const value = useMemo(
    () => ({ mode, resolvedMode, setMode, mounted }),
    [mode, resolvedMode, setMode, mounted],
  );

  return (
    <ThemeModeContext.Provider value={value}>
      {children}
    </ThemeModeContext.Provider>
  );
}

export function useThemeMode() {
  const ctx = useContext(ThemeModeContext);
  if (!ctx)
    throw new Error("useThemeMode must be used within ThemeModeProvider");
  return ctx;
}
