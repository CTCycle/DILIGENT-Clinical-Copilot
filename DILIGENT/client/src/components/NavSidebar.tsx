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
        <rect x="2" y="2" width="32" height="32" rx="8" fill="rgba(22, 163, 74, 0.24)" />
        <path
            d="M10 26V10h6.5c6 0 9.5 3.1 9.5 8s-3.5 8-9.5 8H10zm5-4h1.2c3.2 0 5-1.3 5-4s-1.8-4-5-4H15v8z"
            fill="#4ade80"
        />
        <text x="42" y="16" fill="rgba(255,255,255,0.92)" fontSize="10" fontWeight="700" letterSpacing="0.14em">
            DILIGENT
        </text>
        <text x="42" y="27" fill="rgba(255,255,255,0.62)" fontSize="7.5" letterSpacing="0.08em">
            CLINICAL COPILOT
        </text>
    </svg>
);

const AgentIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        <line x1="9" y1="10" x2="15" y2="10" />
    </svg>
);

const DataIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="6" rx="7" ry="3" />
        <path d="M5 6v6c0 1.66 3.13 3 7 3s7-1.34 7-3V6" />
        <path d="M5 12v6c0 1.66 3.13 3 7 3s7-1.34 7-3v-6" />
    </svg>
);

const SettingsIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
            className={`nav-item ${isActive ? "active" : ""}`}
            onClick={() => onClick(pageId)}
            aria-label={label}
            title={label}
        >
            <span className="nav-icon">{icon}</span>
        </button>
    );
}

// ---------------------------------------------------------------------------
// NavSidebar
// ---------------------------------------------------------------------------
interface NavSidebarProps {
    onNavigate: (pageId: PageId) => void;
}

export function NavSidebar({ onNavigate }: NavSidebarProps): React.JSX.Element {
    const { state } = useAppState();

    const navItems: { pageId: PageId; icon: React.ReactNode; label: string }[] = [
        { pageId: "dili-agent", icon: <AgentIcon />, label: "DILI Agent" },
        { pageId: "data-inspection", icon: <DataIcon />, label: "Data Inspection" },
    ];

    return (
        <nav className="nav-sidebar" aria-label="Main navigation">
            <div className="nav-brand">
                <span className="nav-brand-logo">
                    <DiligentLogo />
                </span>
            </div>
            <div className="nav-items">
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
            <div className="nav-footer">
                <button
                    type="button"
                    className={`nav-item nav-settings ${state.activePage === "model-config" ? "active" : ""}`}
                    onClick={() => onNavigate("model-config")}
                    aria-label="Model Configurations"
                    title="Model Configurations"
                >
                    <span className="nav-icon">
                        <SettingsIcon />
                    </span>
                </button>
            </div>
        </nav>
    );
}
