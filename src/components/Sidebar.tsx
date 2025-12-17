import { Plus, Trash2, Volume2, Moon, Sun } from "lucide-react";

interface Speaker {
    id: string;
    name: string;
    voice: string;
    color: string;
}

interface SidebarProps {
    speakers: Speaker[];
    voices: string[];
    onAddSpeaker: () => void;
    onUpdateSpeaker: (id: string, updates: Partial<Speaker>) => void;
    onDeleteSpeaker: (id: string) => void;
    modelType: string | null;
    theme: 'light' | 'dark';
    onThemeToggle: () => void;
}

export function Sidebar({
    speakers,
    voices,
    onAddSpeaker,
    onUpdateSpeaker,
    onDeleteSpeaker,
    modelType,
    theme,
    onThemeToggle,
}: SidebarProps) {
    const isDark = theme === 'dark';

    return (
        <aside className={`fixed left-0 top-0 w-80 h-screen flex flex-col border-r ${isDark
            ? 'bg-zinc-900 border-zinc-800'
            : 'bg-white border-gray-200'
            }`}>
            <div className={`p-6 border-b ${isDark ? 'border-zinc-800' : 'border-gray-200'}`}>
                <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-teal-600 rounded-lg">
                            <Volume2 className="w-5 h-5 text-white" />
                        </div>
                        <h1 className={`text-xl font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                            VibeVoice Podcast Generator
                        </h1>
                    </div>

                    <button
                        onClick={onThemeToggle}
                        className={`p-2 rounded-lg transition-colors ${isDark
                            ? 'hover:bg-zinc-800 text-zinc-400 hover:text-white'
                            : 'hover:bg-gray-100 text-gray-600 hover:text-gray-900'
                            }`}
                        title={`Switch to ${isDark ? 'light' : 'dark'} mode`}
                    >
                        {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                    </button>
                </div>

                <p className={`text-sm ${isDark ? 'text-zinc-400' : 'text-gray-600'}`}>
                    Create multi-voice podcasts with AI
                </p>

                {modelType && (
                    <div className="mt-3">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${modelType === '0.5B'
                            ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                            : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                            }`}>
                            {modelType === '0.5B' ? '⚡ Realtime 0.5B' : '🎙️ Advanced 1.5B'}
                        </span>
                    </div>
                )}
            </div>

            <div className="flex-1 overflow-y-auto p-6">
                <div className="flex items-center justify-between mb-4">
                    <h2 className={`text-sm font-semibold uppercase tracking-wide ${isDark ? 'text-white' : 'text-gray-900'
                        }`}>
                        Speakers
                    </h2>
                    <button
                        onClick={onAddSpeaker}
                        className="p-1.5 bg-teal-600 hover:bg-teal-700 rounded-lg transition-colors"
                        title="Add Speaker"
                    >
                        <Plus className="w-4 h-4 text-white" />
                    </button>
                </div>

                <div className="space-y-3">
                    {speakers.map((speaker) => (
                        <div
                            key={speaker.id}
                            className={`rounded-xl p-4 space-y-3 border ${isDark
                                ? 'bg-zinc-800 border-zinc-700'
                                : 'bg-gray-50 border-gray-200'
                                }`}
                        >
                            <div className="flex items-center gap-2">
                                <div
                                    className="w-3 h-3 rounded-full flex-shrink-0"
                                    style={{ backgroundColor: speaker.color }}
                                />
                                <input
                                    type="text"
                                    value={speaker.name}
                                    onChange={(e) =>
                                        onUpdateSpeaker(speaker.id, { name: e.target.value })
                                    }
                                    className={`flex-1 rounded-lg px-3 py-1.5 text-sm border focus:outline-none focus:ring-2 focus:ring-teal-500 ${isDark
                                        ? 'bg-zinc-900 border-zinc-700 text-white'
                                        : 'bg-white border-gray-300 text-gray-900'
                                        }`}
                                    placeholder="Speaker name"
                                />
                                <button
                                    onClick={() => onDeleteSpeaker(speaker.id)}
                                    className={`p-1.5 rounded-lg transition-colors ${isDark
                                        ? 'text-zinc-400 hover:text-red-400 hover:bg-zinc-700'
                                        : 'text-gray-400 hover:text-red-600 hover:bg-gray-100'
                                        }`}
                                    title="Delete Speaker"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            </div>

                            <select
                                value={speaker.voice}
                                onChange={(e) =>
                                    onUpdateSpeaker(speaker.id, { voice: e.target.value })
                                }
                                className={`w-full rounded-lg px-3 py-2 text-sm border focus:outline-none focus:ring-2 focus:ring-teal-500 ${isDark
                                    ? 'bg-zinc-900 border-zinc-700 text-white'
                                    : 'bg-white border-gray-300 text-gray-900'
                                    }`}
                            >
                                {voices.map((voice) => (
                                    <option key={voice} value={voice}>
                                        {voice}
                                    </option>
                                ))}
                            </select>
                        </div>
                    ))}

                    {speakers.length === 0 && (
                        <div className={`text-center py-8 text-sm ${isDark ? 'text-zinc-500' : 'text-gray-400'
                            }`}>
                            No speakers yet. Click + to add one.
                        </div>
                    )}
                </div>
            </div>

            <div className={`p-4 border-t text-xs ${isDark
                ? 'border-zinc-800 text-zinc-500'
                : 'border-gray-200 text-gray-500'
                }`}>
                <p>💡 Tip: Assign speakers to segments in the main area</p>
            </div>
        </aside>
    );
}
