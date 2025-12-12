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

const DatabaseIcon = () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
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
export function NavSidebar(): React.JSX.Element {
    const { state, setActivePage } = useAppState();

    const navItems: { pageId: PageId; icon: React.ReactNode; label: string }[] = [
        { pageId: "dili-agent", icon: <AgentIcon />, label: "DILI Agent" },
        { pageId: "database-browser", icon: <DatabaseIcon />, label: "Database Browser" },
    ];

    return (
        <nav className="nav-sidebar" aria-label="Main navigation">
            <div className="nav-brand">
                <span className="nav-brand-letter">DILI</span>
            </div>
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
        </nav>
    );
}
