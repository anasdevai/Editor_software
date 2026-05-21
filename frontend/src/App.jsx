import React, { Suspense, lazy } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import MainLayout from './layouts/MainLayout'
import DashboardPage from './pages/DashboardPage'
import SOPsPage from './pages/SOPsPage'
import KnowledgePage from './pages/KnowledgePage'
import ChatPage from './pages/ChatPage'

import EntitiesPage from './pages/EntitiesPage'
const EditorPage = lazy(() => import('./pages/EditorPage'))

// Placeholder for other pages
const UnderConstruction = ({ title }) => (
  <div style={{ padding: '40px', textAlign: 'center' }}>
    <h2 style={{ fontFamily: 'Inria Serif, serif', color: '#357856' }}>{title}</h2>
    <p>Diese Seite befindet sich noch im Aufbau.</p>
  </div>
)

/**
 * App.jsx
 * 
 * Central Router for the Cybrain Quality System.
 * Managed routing between the Dashboard, SOP list, and the specialized Editor.
 */
function App() {
  return (
    <Router>
      <Routes>
        {/* Main Application Shell */}
        <Route path="/" element={<MainLayout />}>
          <Route index element={<DashboardPage />} />
          <Route path="dashboard" element={<Navigate to="/" replace />} />

          <Route path="knowledge" element={<KnowledgePage />} />
          <Route path="chat" element={<ChatPage />} />
          <Route path="sops" element={<SOPsPage />} />

          <Route path="deviations" element={<EntitiesPage type="deviations" />} />
          <Route path="capa" element={<EntitiesPage type="capas" />} />
          <Route path="audits" element={<EntitiesPage type="audits" />} />
          <Route path="decisions" element={<EntitiesPage type="decisions" />} />

          <Route path="settings" element={<UnderConstruction title="Einstellungen" />} />
          <Route path="help" element={<UnderConstruction title="Helfen" />} />
        </Route>

        {/* Specialized Editor Route - Can be standalone or within layout */}
        {/* For now, we keep it standalone as the legacy editor is very complex */}
        <Route
          path="/editor"
          element={
            <Suspense fallback={<div style={{ padding: '24px', color: '#667085' }}>Lade Editor...</div>}>
              <EditorPage />
            </Suspense>
          }
        />
        <Route
          path="/editor/:id"
          element={
            <Suspense fallback={<div style={{ padding: '24px', color: '#667085' }}>Lade Editor...</div>}>
              <EditorPage />
            </Suspense>
          }
        />

        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  )
}

export default App