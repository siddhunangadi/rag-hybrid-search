import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { Link, useLocation } from 'react-router-dom';
import { BarChart3, Upload, MessageCircle, BookOpen, History, Activity, Settings } from 'lucide-react';
export default function Layout({ children }) {
    const location = useLocation();
    const isActive = (path) => location.pathname === path;
    const navItems = [
        { href: '/', icon: BarChart3, label: 'Dashboard' },
        { href: '/upload', icon: Upload, label: 'Upload' },
        { href: '/chat', icon: MessageCircle, label: 'AI Assistant' },
        { href: '/regulations', icon: BookOpen, label: 'Regulations' },
        { href: '/audit', icon: History, label: 'Audit' },
        { href: '/health', icon: Activity, label: 'Health' },
        { href: '/admin', icon: Settings, label: 'Admin' },
    ];
    return (_jsxs("div", { className: "flex h-screen bg-slate-950", children: [_jsxs("aside", { className: "w-64 border-r border-slate-800 bg-slate-900 flex flex-col", children: [_jsx("div", { className: "p-6 border-b border-slate-800", children: _jsxs("div", { className: "flex items-center gap-3", children: [_jsx("div", { className: "w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg" }), _jsxs("div", { children: [_jsx("h1", { className: "font-semibold", children: "RAG Search" }), _jsx("p", { className: "text-xs text-slate-400", children: "Compliance AI" })] })] }) }), _jsx("nav", { className: "flex-1 px-4 py-6 space-y-1 overflow-y-auto", children: navItems.map((item) => (_jsxs(Link, { to: item.href, className: `flex items-center gap-3 px-4 py-2 rounded-lg transition-all ${isActive(item.href)
                                ? 'bg-blue-600 text-white font-medium'
                                : 'text-slate-300 hover:bg-slate-800'}`, children: [_jsx(item.icon, { className: "w-4 h-4" }), _jsx("span", { className: "text-sm", children: item.label })] }, item.href))) }), _jsx("div", { className: "p-4 border-t border-slate-800 space-y-2", children: _jsx("p", { className: "text-xs text-slate-400", children: "Enterprise v0.1.0" }) })] }), _jsx("main", { className: "flex-1 overflow-auto", children: _jsx("div", { className: "max-w-7xl mx-auto px-8 py-8", children: children }) })] }));
}
