import React from "react";
import { PageId, useAppState } from "../context/AppStateContext";

// ---------------------------------------------------------------------------
// Icon Components
// ---------------------------------------------------------------------------
const DiligentLogo = () => (
    <svg
        viewBox="0 0 120 36"
        role="img"
        aria-labelledby="diligent-logo-title"
    >
        <title id="diligent-logo-title">Application logo</title>
        <defs>
            <linearGradient id="diligent-mark-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#1f9d76" />
                <stop offset="100%" stopColor="#14945f" />
            </linearGradient>
        </defs>
        <rect x="2" y="2" width="32" height="32" rx="9" fill="url(#diligent-mark-gradient)" />
        <rect x="2" y="2" width="32" height="32" rx="9" fill="none" stroke="rgba(15, 23, 42, 0.08)" />
        <path
            d="M10 26V10h6.5c6 0 9.5 3.1 9.5 8s-3.5 8-9.5 8H10zm5-4h1.2c3.2 0 5-1.3 5-4s-1.8-4-5-4H15v8z"
            fill="#effcf6"
        />
        <text x="42" y="16" fill="currentColor" fontSize="10.2" fontWeight="800" letterSpacing="0.13em">
            DILIGENT
        </text>
        <text x="42" y="27" fill="currentColor" fillOpacity="0.74" fontSize="7.4" fontWeight="600" letterSpacing="0.09em">
            CLINICAL COPILOT
        </text>
    </svg>
);

const AgentIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        <line x1="9" y1="10" x2="15" y2="10" />
    </svg>
);

const DataIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <ellipse cx="12" cy="6" rx="7" ry="3" />
        <path d="M5 6v6c0 1.66 3.13 3 7 3s7-1.34 7-3V6" />
        <path d="M5 12v6c0 1.66 3.13 3 7 3s7-1.34 7-3v-6" />
    </svg>
);

const SettingsIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
);

// ---------------------------------------------------------------------------
// NavItem
// ---------------------------------------------------------------------------
interface NavItemProps {
    pageId: PageId;
    icon: React.ReactNode;
    label: string;
    isActive: boolean;
    onClick: (pageId: PageId) => void;
}

function NavItem({ pageId, icon, label, isActive, onClick }: NavItemProps): React.JSX.Element {
    return (
        <button
            type="button"
            className={`tab-item ${isActive ? "active" : ""}`}
            onClick={() => onClick(pageId)}
            aria-current={isActive ? "page" : undefined}
            title={label}
        >
            <span className="tab-icon">{icon}</span>
            <span className="tab-label">{label}</span>
        </button>
    );
}

// ---------------------------------------------------------------------------
// NavTabs
// ---------------------------------------------------------------------------
interface NavTabsProps {
    onNavigate: (pageId: PageId) => void;
}

export function NavTabs({ onNavigate }: NavTabsProps): React.JSX.Element {
    const { state, toggleTheme } = useAppState();

    const navItems: { pageId: PageId; icon: React.ReactNode; label: string }[] = [
        { pageId: "dili-agent", icon: <AgentIcon />, label: "DILI Agent" },
        { pageId: "data-inspection", icon: <DataIcon />, label: "Data Inspection" },
        { pageId: "model-config", icon: <SettingsIcon />, label: "Model Configurations" },
    ];

    return (
        <div className="app-top-nav">
            <header className="app-header">
                <div className="app-header-content">
                    <div className="app-header-brand">
                        <span className="app-header-logo">
                            <DiligentLogo />
                        </span>
                    </div>
                    <nav className="tab-bar" aria-label="Main navigation">
                        <div className="tab-items">
                            {navItems.map((item) => (
                                <NavItem
                                    key={item.pageId}
                                    pageId={item.pageId}
                                    icon={item.icon}
                                    label={item.label}
                                    isActive={state.activePage === item.pageId}
                                    onClick={onNavigate}
                                />
                            ))}
                        </div>
                    </nav>
                    <div className="app-header-actions">
                        <button type="button" className="app-header-icon-btn" aria-label="Help">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <circle cx="12" cy="12" r="10" />
                                <path d="M9.09 9a3 3 0 0 1 5.82 1c0 2-3 3-3 3" />
                                <line x1="12" y1="17" x2="12.01" y2="17" />
                            </svg>
                        </button>
                        <button
                            type="button"
                            className="app-header-icon-btn"
                            onClick={toggleTheme}
                            aria-label="Toggle dark mode"
                            title="Toggle dark mode"
                        >
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M21 12.79A9 9 0 1 1 11.21 3c0 .28-.01.56-.01.84a8 8 0 0 0 9.8 7.95z" />
                            </svg>
                        </button>
                        <span className="app-header-avatar" aria-hidden="true">DR</span>
                    </div>
                </div>
            </header>
        </div>
    );
}

