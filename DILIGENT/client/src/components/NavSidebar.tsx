import React from "react";
import { PageId, useAppState } from "../context/AppStateContext";

// ---------------------------------------------------------------------------
// Icon Components
// ---------------------------------------------------------------------------
const AgentIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
        <line x1="9" y1="10" x2="15" y2="10" />
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
    onOpenConfigModal: () => void;
}

export function NavSidebar({ onOpenConfigModal }: NavSidebarProps): React.JSX.Element {
    const { state, setActivePage } = useAppState();

    const navItems: { pageId: PageId; icon: React.ReactNode; label: string }[] = [
        { pageId: "dili-agent", icon: <AgentIcon />, label: "DILI Agent" },
    ];

    return (
        <nav className="nav-sidebar" aria-label="Main navigation">
            <div className="nav-brand" aria-hidden="true" />
            <div className="nav-items">
                {navItems.map((item) => (
                    <NavItem
                        key={item.pageId}
                        pageId={item.pageId}
                        icon={item.icon}
                        label={item.label}
                        isActive={state.activePage === item.pageId}
                        onClick={setActivePage}
                    />
                ))}
            </div>
            {state.activePage === "dili-agent" && (
                <div className="nav-footer">
                    <button
                        type="button"
                        className="nav-item nav-settings"
                        onClick={onOpenConfigModal}
                        aria-label="Model Configurations"
                        title="Model Configurations"
                    >
                        <span className="nav-icon">
                            <SettingsIcon />
                        </span>
                    </button>
                </div>
            )}
        </nav>
    );
}
