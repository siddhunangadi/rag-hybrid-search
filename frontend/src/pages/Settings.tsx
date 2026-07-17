import { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings as SettingsIcon, Bell, Lock, Palette } from 'lucide-react'
import { Card, Button, Input } from '@/components/ui'

export default function Settings() {
  const [emailNotifications, setEmailNotifications] = useState(true)
  const [apiAlerts, setApiAlerts] = useState(true)
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  return (
    <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="max-w-2xl space-y-6">
      <div className="flex items-center gap-2">
        <SettingsIcon className="w-6 h-6 text-blue-400" />
        <div>
          <h1 className="text-3xl font-bold">Settings</h1>
          <p className="text-slate-400 mt-1">Manage your preferences and account settings</p>
        </div>
      </div>

      {/* Notifications */}
      <Card>
        <div className="flex items-start gap-3 mb-4">
          <Bell className="w-5 h-5 text-amber-400 mt-1" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold">Notifications</h3>
            <p className="text-sm text-slate-400 mt-1">Configure how you receive alerts</p>
          </div>
        </div>

        <div className="space-y-4 pt-4 border-t border-slate-700">
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={emailNotifications}
              onChange={(e) => setEmailNotifications(e.target.checked)}
              className="w-4 h-4 rounded border-slate-600 bg-slate-800 accent-blue-600"
            />
            <span className="text-sm">Email notifications for compliance changes</span>
          </label>

          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={apiAlerts}
              onChange={(e) => setApiAlerts(e.target.checked)}
              className="w-4 h-4 rounded border-slate-600 bg-slate-800 accent-blue-600"
            />
            <span className="text-sm">API alerts and system errors</span>
          </label>
        </div>
      </Card>

      {/* Display */}
      <Card>
        <div className="flex items-start gap-3 mb-4">
          <Palette className="w-5 h-5 text-purple-400 mt-1" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold">Display</h3>
            <p className="text-sm text-slate-400 mt-1">Customize appearance</p>
          </div>
        </div>

        <div className="space-y-4 pt-4 border-t border-slate-700">
          <div>
            <label className="text-sm font-medium text-slate-300 block mb-2">Theme</label>
            <div className="flex gap-2">
              {(['dark', 'light'] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => setTheme(t)}
                  className={`px-4 py-2 rounded text-sm font-medium transition-all ${
                    theme === t
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                  }`}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Card>

      {/* Security */}
      <Card>
        <div className="flex items-start gap-3 mb-4">
          <Lock className="w-5 h-5 text-red-400 mt-1" />
          <div className="flex-1">
            <h3 className="text-lg font-semibold">Security</h3>
            <p className="text-sm text-slate-400 mt-1">Manage authentication and access</p>
          </div>
        </div>

        <div className="space-y-4 pt-4 border-t border-slate-700">
          <div>
            <label className="text-sm font-medium text-slate-300 block mb-2">API Key</label>
            <div className="flex gap-2">
              <Input type="password" value="••••••••••••••••" readOnly className="flex-1" />
              <Button variant="secondary">Regenerate</Button>
            </div>
            <p className="text-xs text-slate-400 mt-2">Last regenerated: 30 days ago</p>
          </div>

          <div className="pt-4 border-t border-slate-700">
            <h4 className="text-sm font-medium text-slate-300 mb-3">Active Sessions</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                <div>
                  <p className="text-sm font-medium">Current Session</p>
                  <p className="text-xs text-slate-400">Browser - Active now</p>
                </div>
                <span className="text-xs text-emerald-400 font-medium">Active</span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Save */}
      <div className="flex gap-3">
        <Button>Save Changes</Button>
        <Button variant="secondary">Cancel</Button>
      </div>
    </motion.div>
  )
}
