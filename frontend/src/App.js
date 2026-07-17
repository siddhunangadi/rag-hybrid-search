import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Upload from './pages/Upload';
import Chat from './pages/Chat';
import Regulations from './pages/Regulations';
import Audit from './pages/Audit';
import Health from './pages/Health';
import Admin from './pages/Admin';
export default function App() {
    return (_jsx(Router, { children: _jsx(Layout, { children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(Dashboard, {}) }), _jsx(Route, { path: "/upload", element: _jsx(Upload, {}) }), _jsx(Route, { path: "/chat", element: _jsx(Chat, {}) }), _jsx(Route, { path: "/regulations", element: _jsx(Regulations, {}) }), _jsx(Route, { path: "/audit", element: _jsx(Audit, {}) }), _jsx(Route, { path: "/health", element: _jsx(Health, {}) }), _jsx(Route, { path: "/admin", element: _jsx(Admin, {}) })] }) }) }));
}
