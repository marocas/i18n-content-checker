"use client";

import DarkModeIcon from "@mui/icons-material/DarkMode";
import LightModeIcon from "@mui/icons-material/LightMode";
import SettingsBrightnessIcon from "@mui/icons-material/SettingsBrightness";
import { IconButton, Tooltip } from "@mui/material";
import { useThemeMode } from "./ThemeModeContext";

type ThemeMode = "light" | "dark" | "system";

const MODE_CYCLE: ThemeMode[] = ["light", "dark", "system"];

const MODE_CONFIG: Record<ThemeMode, { icon: React.ReactNode; label: string }> =
  {
    light: { icon: <LightModeIcon />, label: "Light mode" },
    dark: { icon: <DarkModeIcon />, label: "Dark mode" },
    system: { icon: <SettingsBrightnessIcon />, label: "System preference" },
  };

export default function ThemeModeToggle() {
  const { mode, setMode } = useThemeMode();

  const handleToggle = () => {
    const currentIndex = MODE_CYCLE.indexOf(mode);
    const nextIndex = (currentIndex + 1) % MODE_CYCLE.length;
    setMode(MODE_CYCLE[nextIndex]);
  };

  const { icon, label } = MODE_CONFIG[mode];

  return (
    <Tooltip title={label}>
      <IconButton
        suppressHydrationWarning={true}
        onClick={handleToggle}
        color="inherit"
        aria-label={label}
      >
        {icon}
      </IconButton>
    </Tooltip>
  );
}
