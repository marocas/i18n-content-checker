"use client";

import { darkTheme, lightTheme } from "@/theme";
import { CssBaseline, ThemeProvider } from "@mui/material";
import { useMemo } from "react";
import { ThemeModeProvider, useThemeMode } from "./ThemeModeContext";

function ThemedApp({ children }: { children: React.ReactNode }) {
  const { resolvedMode, mounted } = useThemeMode();
  const theme = useMemo(
    () => (resolvedMode === "dark" ? darkTheme : lightTheme),
    [resolvedMode],
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <div style={{ visibility: mounted ? "visible" : "hidden" }}>
        {children}
      </div>
    </ThemeProvider>
  );
}

export default function ThemeRegistry({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ThemeModeProvider>
      <ThemedApp>{children}</ThemedApp>
    </ThemeModeProvider>
  );
}
