import React from "react";
import { AppStateProvider, useAppState } from "./context/AppStateContext";
import { NavSidebar } from "./components/NavSidebar";
import { DiluAgentPage } from "./pages/DiluAgentPage";
import { DatabaseBrowserPage } from "./pages/DatabaseBrowserPage";

// ---------------------------------------------------------------------------
// AppContent - Renders the active page
// ---------------------------------------------------------------------------
function AppContent(): React.JSX.Element {
  const { state } = useAppState();

  return (
    <div className="app-shell">
      <NavSidebar />
      <div className="app-main">
        {state.activePage === "dili-agent" && <DiluAgentPage />}
        {state.activePage === "database-browser" && <DatabaseBrowserPage />}
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
