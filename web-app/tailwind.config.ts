import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        slate: {
          950: "#0f1729",
        },
        primary: {
          50: "#e6f0ff",
          100: "#cce0ff",
          200: "#99c2ff",
          300: "#66a3ff",
          400: "#3385ff",
          500: "#0066ff",
          600: "#0052cc",
          700: "#003d99",
          800: "#002966",
          900: "#001433",
        },
        success: {
          500: "#0db678",
        },
        warning: {
          500: "#f59e0b",
        },
        danger: {
          500: "#ef4444",
        },
      },
      fontFamily: {
        display: ["'Inter var'", "Inter", "system-ui", "sans-serif"],
        mono: ["'JetBrains Mono'", "monospace"],
      },
      boxShadow: {
        card: "0 18px 30px -25px rgba(15, 23, 42, 0.55)",
        inset: "inset 0 1px 0 rgba(255, 255, 255, 0.06)",
      },
      borderRadius: {
        "2xl": "1.25rem",
      },
      scale: {
        102: "1.02",
      },
    },
  },
  plugins: [],
};

export default config;
