import { useCallback, useEffect, useRef, useState } from "react";
import HomePage from "./pages/HomePage";
import { BackendStatus } from "./components/Header";
import "./styles/global.css";

type Theme = "light" | "dark";

const ALL_PAGE_IDS = ["page-upload", "page-results", "page-how"];
const ALL_PAGE_NAMES = ["Home", "Results", "How It Works"];

function getSystemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

function getStoredTheme(): Theme | null {
  try {
    const v = localStorage.getItem("theme");
    if (v === "light" || v === "dark") return v;
  } catch {
    /* ignore */
  }
  return null;
}

export default function App() {
  const [theme, setTheme] = useState<Theme>(
    () => getStoredTheme() ?? getSystemTheme(),
  );
  const [activePage, setActivePage] = useState(0);
  const [denoiseComplete, setDenoiseComplete] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const pageIds = denoiseComplete
    ? ALL_PAGE_IDS
    : [ALL_PAGE_IDS[0], ALL_PAGE_IDS[2]];
  const pageNames = denoiseComplete
    ? ALL_PAGE_NAMES
    : [ALL_PAGE_NAMES[0], ALL_PAGE_NAMES[2]];

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = (e: MediaQueryListEvent) => {
      if (!getStoredTheme()) setTheme(e.matches ? "dark" : "light");
    };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  /* Track which page is visible via scroll position */
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const handleScroll = () => {
      const scrollTop = container.scrollTop;
      const sectionHeight = container.clientHeight;
      const page = Math.round(scrollTop / sectionHeight);
      setActivePage(Math.max(0, Math.min(page, pageIds.length - 1)));
    };
    container.addEventListener("scroll", handleScroll, { passive: true });
    return () => container.removeEventListener("scroll", handleScroll);
  }, [pageIds.length]);

  const toggleTheme = () => setTheme((t) => (t === "light" ? "dark" : "light"));

  const scrollToPage = useCallback(
    (index: number) => {
      const section = document.getElementById(pageIds[index]);
      section?.scrollIntoView({ behavior: "smooth" });
    },
    [pageIds],
  );

  return (
    <>
      <div className="grain" aria-hidden="true" />

      {/* ── Permanent navbar ── */}
      <header className="header">
        <div className="header__brand">
          <span className="header__brand-icon" aria-hidden="true">
            <svg
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M9 18V5l12-2v13" />
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="16" r="3" />
            </svg>
          </span>
          <span className="header__title">DenoiseAI</span>
          <span className="header__separator">|</span>
          <button className="header__home-link" onClick={() => scrollToPage(0)}>
            Home
          </button>
        </div>

        <div className="header__actions">
          <BackendStatus />
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            <span className="theme-toggle__thumb" />
          </button>
        </div>
      </header>

      {/* ── Left side-nav ── */}
      <nav className="side-nav" aria-label="Page navigation">
        {pageNames.map((name, i) => (
          <button
            key={pageIds[i]}
            className={`side-nav__item ${i === activePage ? "side-nav__item--active" : ""}`}
            onClick={() => scrollToPage(i)}
          >
            <span className="side-nav__dot" />
            <span className="side-nav__label">{name}</span>
          </button>
        ))}
      </nav>

      {/* ── Fullpage scroll container ── */}
      <div className="fullpage-container" ref={containerRef}>
        <HomePage onDenoiseComplete={() => setDenoiseComplete(true)} />
      </div>
    </>
  );
}
