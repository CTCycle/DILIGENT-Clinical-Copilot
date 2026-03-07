import React, { useEffect } from "react";
import {
  AppStateProvider,
  PageId,
  resolvePageIdFromPath,
  resolvePathFromPage,
  useAppState,
} from "./context/AppStateContext";
import { NavTabs } from "./components/NavSidebar";
import { DiliAgentPage } from "./pages/DiliAgentPage";
import { DataInspectionPage } from "./pages/DataInspectionPage";
import { ModelConfigPage } from "./pages/ModelConfigPage";

// ---------------------------------------------------------------------------
// AppContent - Renders the active page
// ---------------------------------------------------------------------------
function AppContent(): React.JSX.Element {
  const { state, setActivePage } = useAppState();

  useEffect(() => {
    const handlePopState = () => {
      setActivePage(resolvePageIdFromPath(window.location.pathname));
    };
    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("popstate", handlePopState);
    };
  }, [setActivePage]);

  const navigateToPage = (pageId: PageId) => {
    const nextPath = resolvePathFromPage(pageId);
    if (window.location.pathname !== nextPath) {
      window.history.pushState({}, "", nextPath);
    }
    setActivePage(pageId);
  };

  return (
    <div className="app-shell">
      <NavTabs onNavigate={navigateToPage} />
      <div className="app-main">
        {state.activePage === "dili-agent" && <DiliAgentPage />}
        {state.activePage === "data-inspection" && <DataInspectionPage />}
        {state.activePage === "model-config" && <ModelConfigPage />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// App - Root component with state provider
// ---------------------------------------------------------------------------
function App(): React.JSX.Element {
  return (
    <AppStateProvider>
      <AppContent />
    </AppStateProvider>
  );
}

export default App;

