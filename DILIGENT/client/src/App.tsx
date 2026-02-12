import React, { useState } from "react";
import { AppStateProvider, useAppState } from "./context/AppStateContext";
import { NavSidebar } from "./components/NavSidebar";
import { DiluAgentPage } from "./pages/DiluAgentPage";

// ---------------------------------------------------------------------------
// AppContent - Renders the active page
// ---------------------------------------------------------------------------
function AppContent(): React.JSX.Element {
  const { state } = useAppState();
  const [configModalOpen, setConfigModalOpen] = useState(false);

  return (
    <div className="app-shell">
      <NavSidebar onOpenConfigModal={() => setConfigModalOpen(true)} />
      <div className="app-main">
        {state.activePage === "dili-agent" && (
          <DiluAgentPage
            configModalOpen={configModalOpen}
            onCloseConfigModal={() => setConfigModalOpen(false)}
          />
        )}
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
